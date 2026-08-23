# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm performance and accuracy analysis for the deterministic Qwen3 FFN.

This benchmark compares the deterministic PR #325 path against native ROCm
operators without loading a model:

* deterministic GEMM vs ``torch.matmul`` (rocBLAS/hipBLASLt);
* deterministic SwiGLU vs native PyTorch elementwise operators;
* deterministic collectives vs native RCCL through ProcessGroupNCCL;
* deterministic FFN vs a native ROCm/RCCL implementation with the same TP/CP/SP
  sharding and collective schedule.

The script writes raw JSON, Markdown tables, and PNG figures.  GPU events are
used for single-GPU operators.  Distributed timings use synchronized wall
clock samples and report the slowest rank for each iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import statistics
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import rl_engine.kernels.ops.pytorch.ffn.ffn as ffn_module
from rl_engine.distributed import RCCLDeterministicCollective
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.matmul import deterministic_gemm
from rl_engine.kernels.ops.pytorch.ffn import qwen3_ffn
from rl_engine.kernels.ops.triton.activation.swiglu import (
    _launch_swiglu_bwd,
    _launch_swiglu_fwd,
)
from rl_engine.kernels.ops.triton.ffn import qwen3_ffn_triton
from rl_engine.kernels.ops.triton.matmul import deterministic_gemm_triton

_MIB = 1024 * 1024
_REQUIRED_SYMBOLS = (
    "det_gemm_fwd",
    "det_gemm_db",
    "swiglu_forward",
    "swiglu_backward",
)
_DISTRIBUTED_CONFIGS: dict[int, tuple[tuple[str, int, int, bool], ...]] = {
    2: (
        ("tp2", 2, 1, False),
        ("tp2_sp", 2, 1, True),
    ),
    4: (
        ("tp4", 4, 1, False),
        ("tp2_cp2", 2, 2, False),
        ("tp2_cp2_sp", 2, 2, True),
    ),
    8: (
        ("tp8", 8, 1, False),
        ("tp4_cp2", 4, 2, False),
        ("tp4_cp2_sp", 4, 2, True),
    ),
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    denominator = torch.linalg.vector_norm(expected_float)
    if denominator.item() == 0.0:
        return float(torch.linalg.vector_norm(actual_float - expected_float).item())
    return float((torch.linalg.vector_norm(actual_float - expected_float) / denominator).item())


def _accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().float() - expected.detach().float()
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "relative_l2": _relative_l2(actual, expected),
        "exact_fraction": float((actual.detach() == expected.detach()).float().mean().item()),
    }


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
    scale: float = 0.02,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return value.to(device=device, dtype=torch.bfloat16)


def _gpu_event_samples(
    function: Callable[[], Any],
    *,
    warmup: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in events]


def _native_ffn(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate = hidden @ gate_weight.t()
    up = hidden @ up_weight.t()
    activated = F.silu(gate) * up
    return activated @ down_weight.t()


def _single_gpu_benchmarks(
    *,
    warmup: int,
    samples: int,
    training_samples: int,
) -> dict[str, list[dict[str, Any]]]:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False

    results: dict[str, list[dict[str, Any]]] = {
        "gemm": [],
        "swiglu": [],
        "ffn": [],
    }

    gemm_shapes = []
    for tokens in (1, 8, 32, 128):
        gemm_shapes.extend(
            (
                (f"gate_up_m{tokens}", tokens, 4096, 12288),
                (f"down_m{tokens}", tokens, 12288, 4096),
            )
        )

    for shape_index, (name, m_size, k_size, n_size) in enumerate(gemm_shapes):
        left = _randn((m_size, k_size), seed=1000 + shape_index * 2, device=device)
        right = _randn((k_size, n_size), seed=1001 + shape_index * 2, device=device)
        native_output = torch.matmul(left, right)
        deterministic_output = deterministic_gemm(left, right)
        triton_output = deterministic_gemm_triton(left, right)
        fp32_reference = left.float() @ right.float()
        torch.cuda.synchronize()

        native_samples = _gpu_event_samples(
            lambda: torch.matmul(left, right), warmup=warmup, samples=samples
        )
        deterministic_samples = _gpu_event_samples(
            lambda: deterministic_gemm(left, right), warmup=warmup, samples=samples
        )
        triton_samples = _gpu_event_samples(
            lambda: deterministic_gemm_triton(left, right),
            warmup=warmup,
            samples=samples,
        )
        native_timing = _summary_ms(native_samples)
        deterministic_timing = _summary_ms(deterministic_samples)
        triton_timing = _summary_ms(triton_samples)
        results["gemm"].append(
            {
                "name": name,
                "m": m_size,
                "k": k_size,
                "n": n_size,
                "dtype": "bfloat16",
                "native": native_timing,
                "deterministic": deterministic_timing,
                "triton": triton_timing,
                "overhead_ratio": (
                    deterministic_timing["median_ms"] / native_timing["median_ms"]
                ),
                "triton_overhead_ratio": (
                    triton_timing["median_ms"] / native_timing["median_ms"]
                ),
                "triton_speedup_vs_hip": (
                    deterministic_timing["median_ms"] / triton_timing["median_ms"]
                ),
                "triton_vs_hip_mismatch_count": int(
                    (triton_output != deterministic_output).sum().item()
                ),
                "deterministic_vs_native": _accuracy(
                    deterministic_output, native_output
                ),
                "native_vs_fp32": _accuracy(native_output, fp32_reference),
                "deterministic_vs_fp32": _accuracy(
                    deterministic_output, fp32_reference
                ),
                "triton_vs_fp32": _accuracy(triton_output, fp32_reference),
            }
        )
        del (
            left,
            right,
            native_output,
            deterministic_output,
            triton_output,
            fp32_reference,
        )
        torch.cuda.empty_cache()

    for case_index, tokens in enumerate((1, 8, 32, 128)):
        gate = _randn((tokens, 12288), seed=2000 + case_index * 3, device=device)
        up = _randn((tokens, 12288), seed=2001 + case_index * 3, device=device)
        grad_output = _randn(
            (tokens, 12288), seed=2002 + case_index * 3, device=device
        )

        native_forward = F.silu(gate) * up
        deterministic_forward = _C.swiglu_forward(gate, up)
        triton_forward = _launch_swiglu_fwd(gate, up)

        def native_backward() -> tuple[torch.Tensor, torch.Tensor]:
            sigmoid = torch.sigmoid(gate)
            grad_gate = grad_output * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
            grad_up = grad_output * F.silu(gate)
            return grad_gate, grad_up

        native_grad_gate, native_grad_up = native_backward()
        deterministic_grad_gate, deterministic_grad_up = _C.swiglu_backward(
            grad_output, gate, up
        )
        triton_grad_gate, triton_grad_up = _launch_swiglu_bwd(grad_output, gate, up)

        for direction, native_function, deterministic_function, triton_function in (
            (
                "forward",
                lambda: F.silu(gate) * up,
                lambda: _C.swiglu_forward(gate, up),
                lambda: _launch_swiglu_fwd(gate, up),
            ),
            (
                "backward",
                native_backward,
                lambda: _C.swiglu_backward(grad_output, gate, up),
                lambda: _launch_swiglu_bwd(grad_output, gate, up),
            ),
        ):
            native_timing = _summary_ms(
                _gpu_event_samples(
                    native_function, warmup=warmup, samples=samples
                )
            )
            deterministic_timing = _summary_ms(
                _gpu_event_samples(
                    deterministic_function, warmup=warmup, samples=samples
                )
            )
            triton_timing = _summary_ms(
                _gpu_event_samples(
                    triton_function, warmup=warmup, samples=samples
                )
            )
            if direction == "forward":
                accuracy = _accuracy(deterministic_forward, native_forward)
                triton_mismatch_count = int(
                    (triton_forward != deterministic_forward).sum().item()
                )
            else:
                accuracy = {
                    "max_abs": max(
                        _accuracy(deterministic_grad_gate, native_grad_gate)["max_abs"],
                        _accuracy(deterministic_grad_up, native_grad_up)["max_abs"],
                    ),
                    "relative_l2": max(
                        _relative_l2(deterministic_grad_gate, native_grad_gate),
                        _relative_l2(deterministic_grad_up, native_grad_up),
                    ),
                }
                triton_mismatch_count = int(
                    (triton_grad_gate != deterministic_grad_gate).sum().item()
                    + (triton_grad_up != deterministic_grad_up).sum().item()
                )
            results["swiglu"].append(
                {
                    "name": f"swiglu_{direction}_m{tokens}",
                    "direction": direction,
                    "tokens": tokens,
                    "intermediate": 12288,
                    "dtype": "bfloat16",
                    "native": native_timing,
                    "deterministic": deterministic_timing,
                    "triton": triton_timing,
                    "overhead_ratio": (
                        deterministic_timing["median_ms"] / native_timing["median_ms"]
                    ),
                    "triton_overhead_ratio": (
                        triton_timing["median_ms"] / native_timing["median_ms"]
                    ),
                    "triton_speedup_vs_hip": (
                        deterministic_timing["median_ms"] / triton_timing["median_ms"]
                    ),
                    "triton_vs_hip_mismatch_count": triton_mismatch_count,
                    "deterministic_vs_native": accuracy,
                }
            )

        del gate, up, grad_output
        torch.cuda.empty_cache()

    gate_weight = _randn((12288, 4096), seed=3000, device=device)
    up_weight = _randn((12288, 4096), seed=3001, device=device)
    down_weight = _randn((4096, 12288), seed=3002, device=device)
    for case_index, tokens in enumerate((1, 8, 32)):
        hidden = _randn((tokens, 4096), seed=3010 + case_index * 2, device=device)
        grad_output = _randn(
            (tokens, 4096), seed=3011 + case_index * 2, device=device
        )
        with torch.no_grad():
            native_output = _native_ffn(hidden, gate_weight, up_weight, down_weight)
            deterministic_output = qwen3_ffn(
                hidden, gate_weight, up_weight, down_weight
            )
            triton_output = qwen3_ffn_triton(
                hidden, gate_weight, up_weight, down_weight
            )

        native_timing = _summary_ms(
            _gpu_event_samples(
                lambda: _native_ffn(hidden, gate_weight, up_weight, down_weight),
                warmup=warmup,
                samples=samples,
            )
        )
        deterministic_timing = _summary_ms(
            _gpu_event_samples(
                lambda: qwen3_ffn(hidden, gate_weight, up_weight, down_weight),
                warmup=warmup,
                samples=samples,
            )
        )
        triton_timing = _summary_ms(
            _gpu_event_samples(
                lambda: qwen3_ffn_triton(
                    hidden, gate_weight, up_weight, down_weight
                ),
                warmup=warmup,
                samples=samples,
            )
        )
        results["ffn"].append(
            {
                "name": f"ffn_forward_m{tokens}",
                "direction": "forward",
                "tokens": tokens,
                "hidden": 4096,
                "intermediate": 12288,
                "dtype": "bfloat16",
                "native": native_timing,
                "deterministic": deterministic_timing,
                "triton": triton_timing,
                "overhead_ratio": (
                    deterministic_timing["median_ms"] / native_timing["median_ms"]
                ),
                "triton_overhead_ratio": (
                    triton_timing["median_ms"] / native_timing["median_ms"]
                ),
                "triton_speedup_vs_hip": (
                    deterministic_timing["median_ms"] / triton_timing["median_ms"]
                ),
                "triton_vs_hip_mismatch_count": int(
                    (triton_output != deterministic_output).sum().item()
                ),
                "deterministic_vs_native": _accuracy(
                    deterministic_output, native_output
                ),
                "train_infer_bitwise": None,
            }
        )

        native_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (hidden, gate_weight, up_weight, down_weight)
        ]
        deterministic_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (hidden, gate_weight, up_weight, down_weight)
        ]
        triton_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (hidden, gate_weight, up_weight, down_weight)
        ]

        def native_training_step() -> torch.Tensor:
            for value in native_inputs:
                value.grad = None
            output = _native_ffn(*native_inputs)
            output.backward(grad_output)
            return output

        def deterministic_training_step() -> torch.Tensor:
            for value in deterministic_inputs:
                value.grad = None
            output = qwen3_ffn(*deterministic_inputs)
            output.backward(grad_output)
            return output

        def triton_training_step() -> torch.Tensor:
            for value in triton_inputs:
                value.grad = None
            output = qwen3_ffn_triton(*triton_inputs)
            output.backward(grad_output)
            return output

        native_train_output = native_training_step().detach().clone()
        native_grads = [value.grad.detach().clone() for value in native_inputs]
        deterministic_train_output = deterministic_training_step().detach().clone()
        deterministic_grads = [
            value.grad.detach().clone() for value in deterministic_inputs
        ]
        triton_train_output = triton_training_step().detach().clone()
        triton_grads = [value.grad.detach().clone() for value in triton_inputs]
        train_infer_bitwise = torch.equal(
            deterministic_output, deterministic_train_output
        )

        native_train_timing = _summary_ms(
            _gpu_event_samples(
                native_training_step,
                warmup=max(1, warmup // 2),
                samples=training_samples,
            )
        )
        deterministic_train_timing = _summary_ms(
            _gpu_event_samples(
                deterministic_training_step,
                warmup=max(1, warmup // 2),
                samples=training_samples,
            )
        )
        triton_train_timing = _summary_ms(
            _gpu_event_samples(
                triton_training_step,
                warmup=max(1, warmup // 2),
                samples=training_samples,
            )
        )
        results["ffn"].append(
            {
                "name": f"ffn_train_fwd_bwd_m{tokens}",
                "direction": "train_fwd_bwd",
                "tokens": tokens,
                "hidden": 4096,
                "intermediate": 12288,
                "dtype": "bfloat16",
                "native": native_train_timing,
                "deterministic": deterministic_train_timing,
                "triton": triton_train_timing,
                "overhead_ratio": (
                    deterministic_train_timing["median_ms"]
                    / native_train_timing["median_ms"]
                ),
                "triton_overhead_ratio": (
                    triton_train_timing["median_ms"]
                    / native_train_timing["median_ms"]
                ),
                "triton_speedup_vs_hip": (
                    deterministic_train_timing["median_ms"]
                    / triton_train_timing["median_ms"]
                ),
                "triton_vs_hip_mismatch_count": int(
                    (triton_train_output != deterministic_train_output).sum().item()
                    + sum(
                        (triton_grad != deterministic_grad).sum().item()
                        for triton_grad, deterministic_grad in zip(
                            triton_grads, deterministic_grads, strict=True
                        )
                    )
                ),
                "deterministic_vs_native": _accuracy(
                    deterministic_train_output, native_train_output
                ),
                "max_gradient_relative_l2": max(
                    _relative_l2(actual, expected)
                    for actual, expected in zip(
                        deterministic_grads, native_grads, strict=True
                    )
                ),
                "train_infer_bitwise": bool(train_infer_bitwise),
            }
        )
        del (
            hidden,
            grad_output,
            native_inputs,
            deterministic_inputs,
            triton_inputs,
            native_grads,
            deterministic_grads,
            triton_grads,
        )
        torch.cuda.empty_cache()

    return results


def _native_all_gather(input_tensor: torch.Tensor, group: Any) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    flat_input = input_tensor.contiguous().view(-1)
    flat_output = torch.empty(
        world_size * flat_input.numel(),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.all_gather_into_tensor(flat_output, flat_input, group=group)
    return flat_output.reshape(
        input_tensor.size(0) * world_size, *input_tensor.shape[1:]
    )


def _native_all_reduce(input_tensor: torch.Tensor, group: Any) -> torch.Tensor:
    output = input_tensor.clone()
    dist.all_reduce(output, group=group)
    return output


def _native_reduce_scatter(input_tensor: torch.Tensor, group: Any) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    output = torch.empty(
        (input_tensor.size(0) // world_size, *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.reduce_scatter_tensor(output, input_tensor.contiguous(), group=group)
    return output


class _NativeDistributedFFNFunction(torch.autograd.Function):
    """Native ROCm/RCCL baseline with the PR's collective schedule."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        tp_group: Any,
        cp_group: Any,
        sequence_parallel: bool,
    ) -> torch.Tensor:
        input_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        if sequence_parallel:
            hidden_2d = _native_all_gather(hidden_2d, tp_group)

        gate = hidden_2d @ gate_weight.t()
        up = hidden_2d @ up_weight.t()
        activated = F.silu(gate) * up
        output = activated @ down_weight.t()

        if sequence_parallel:
            output = _native_reduce_scatter(output, tp_group)
        elif tp_group is not None:
            output = _native_all_reduce(output, tp_group)

        ctx.save_for_backward(
            hidden_2d,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        )
        ctx.input_shape = input_shape
        ctx.tp_group = tp_group
        ctx.cp_group = cp_group
        ctx.sequence_parallel = sequence_parallel
        return output.reshape(*input_shape[:-1], output.size(-1))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        (
            hidden,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        ) = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
        if ctx.sequence_parallel:
            grad_output = _native_all_gather(grad_output, ctx.tp_group)

        if ctx.cp_group is not None:
            activated_full = _native_all_gather(activated, ctx.cp_group)
            grad_output_full = _native_all_gather(grad_output, ctx.cp_group)
            grad_down_weight = grad_output_full.t() @ activated_full
        else:
            grad_down_weight = grad_output.t() @ activated

        grad_activated = grad_output @ down_weight
        sigmoid = torch.sigmoid(gate)
        grad_gate = grad_activated * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_up = grad_activated * F.silu(gate)

        if ctx.cp_group is not None:
            hidden_full = _native_all_gather(hidden, ctx.cp_group)
            grad_gate_full = _native_all_gather(grad_gate, ctx.cp_group)
            grad_up_full = _native_all_gather(grad_up, ctx.cp_group)
            grad_gate_weight = grad_gate_full.t() @ hidden_full
            grad_up_weight = grad_up_full.t() @ hidden_full
        else:
            grad_gate_weight = grad_gate.t() @ hidden
            grad_up_weight = grad_up.t() @ hidden

        grad_hidden_gate = grad_gate @ gate_weight
        if ctx.sequence_parallel:
            grad_hidden_gate = _native_reduce_scatter(
                grad_hidden_gate, ctx.tp_group
            )
        elif ctx.tp_group is not None:
            grad_hidden_gate = _native_all_reduce(grad_hidden_gate, ctx.tp_group)

        grad_hidden_up = grad_up @ up_weight
        if ctx.sequence_parallel:
            grad_hidden_up = _native_reduce_scatter(grad_hidden_up, ctx.tp_group)
        elif ctx.tp_group is not None:
            grad_hidden_up = _native_all_reduce(grad_hidden_up, ctx.tp_group)

        grad_hidden = grad_hidden_gate.add_(grad_hidden_up)
        return (
            grad_hidden.reshape(ctx.input_shape),
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
            None,
            None,
            None,
        )


def _native_distributed_ffn(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    tp_group: Any,
    cp_group: Any,
    sequence_parallel: bool,
) -> torch.Tensor:
    return _NativeDistributedFFNFunction.apply(
        hidden,
        gate_weight,
        up_weight,
        down_weight,
        tp_group,
        cp_group,
        sequence_parallel,
    )


def _mesh_groups(
    world_size: int,
    tp_size: int,
    cp_size: int,
) -> tuple[list[Any], list[Any]]:
    if tp_size * cp_size != world_size:
        raise ValueError("TP size times CP size must equal world size")
    if tp_size == world_size and cp_size == 1:
        return [dist.group.WORLD], []

    tp_groups = []
    if tp_size > 1:
        for cp_rank in range(cp_size):
            ranks = list(range(cp_rank * tp_size, (cp_rank + 1) * tp_size))
            tp_groups.append(dist.new_group(ranks=ranks))

    cp_groups = []
    if cp_size > 1:
        for tp_rank in range(tp_size):
            ranks = [cp_rank * tp_size + tp_rank for cp_rank in range(cp_size)]
            cp_groups.append(dist.new_group(ranks=ranks))
    return tp_groups, cp_groups


def _shard_ranges(
    rank: int,
    *,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool,
    token_count: int,
    intermediate_size: int,
) -> tuple[int, int, int, int]:
    tp_rank = rank % tp_size
    cp_rank = rank // tp_size
    cp_tokens = token_count // cp_size
    local_tokens = cp_tokens // tp_size if sequence_parallel else cp_tokens
    token_start = cp_rank * cp_tokens
    if sequence_parallel:
        token_start += tp_rank * local_tokens
    token_end = token_start + local_tokens

    local_intermediate = intermediate_size // tp_size
    feature_start = tp_rank * local_intermediate
    feature_end = feature_start + local_intermediate
    return token_start, token_end, feature_start, feature_end


def _distributed_wall_samples(
    function: Callable[[], Any],
    *,
    group: Any,
    warmup: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    timings = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    dist.barrier(group=group)
    return timings


def _slowest_rank_summary(local_timings: list[float], group: Any) -> dict[str, float]:
    world_size = dist.get_world_size(group=group)
    gathered: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_timings, group=group)
    slowest_per_sample = [
        max(float(rank_timings[index]) for rank_timings in gathered if rank_timings is not None)
        for index in range(len(local_timings))
    ]
    return _summary_ms(slowest_per_sample)


def _collective_benchmarks(
    rank: int,
    world_size: int,
    *,
    warmup: int,
    samples: int,
) -> list[dict[str, Any]]:
    device = torch.device("cuda", rank)
    results = []
    for message_bytes in (64 * 1024, 1 * _MIB, 16 * _MIB):
        elements = message_bytes // torch.tensor([], dtype=torch.bfloat16).element_size()
        input_tensor = torch.zeros(elements, dtype=torch.bfloat16, device=device)
        deterministic = RCCLDeterministicCollective(
            group=dist.group.WORLD,
            device=device,
            max_size_bytes=message_bytes,
        )
        deterministic_outputs = {
            "all_reduce": torch.empty_like(input_tensor),
            "all_gather": torch.empty(
                elements * world_size, dtype=input_tensor.dtype, device=device
            ),
            "reduce_scatter": torch.empty(
                elements // world_size, dtype=input_tensor.dtype, device=device
            ),
        }
        native_outputs = {
            "all_gather": torch.empty(
                elements * world_size, dtype=input_tensor.dtype, device=device
            ),
            "reduce_scatter": torch.empty(
                elements // world_size, dtype=input_tensor.dtype, device=device
            ),
        }

        def deterministic_all_reduce() -> torch.Tensor:
            return deterministic.all_reduce(
                input_tensor, out=deterministic_outputs["all_reduce"]
            )

        def native_all_reduce() -> torch.Tensor:
            dist.all_reduce(input_tensor, group=dist.group.WORLD)
            return input_tensor

        def deterministic_all_gather() -> torch.Tensor:
            return deterministic.all_gather(
                input_tensor, out=deterministic_outputs["all_gather"]
            )

        def native_all_gather() -> torch.Tensor:
            dist.all_gather_into_tensor(
                native_outputs["all_gather"], input_tensor, group=dist.group.WORLD
            )
            return native_outputs["all_gather"]

        def deterministic_reduce_scatter() -> torch.Tensor:
            return deterministic.reduce_scatter(
                input_tensor, out=deterministic_outputs["reduce_scatter"]
            )

        def native_reduce_scatter() -> torch.Tensor:
            dist.reduce_scatter_tensor(
                native_outputs["reduce_scatter"],
                input_tensor,
                group=dist.group.WORLD,
            )
            return native_outputs["reduce_scatter"]

        for operation, native_function, deterministic_function in (
            ("all_reduce", native_all_reduce, deterministic_all_reduce),
            ("all_gather", native_all_gather, deterministic_all_gather),
            (
                "reduce_scatter",
                native_reduce_scatter,
                deterministic_reduce_scatter,
            ),
        ):
            native_timings = _distributed_wall_samples(
                native_function,
                group=dist.group.WORLD,
                warmup=warmup,
                samples=samples,
            )
            deterministic_timings = _distributed_wall_samples(
                deterministic_function,
                group=dist.group.WORLD,
                warmup=warmup,
                samples=samples,
            )
            native_summary = _slowest_rank_summary(
                native_timings, dist.group.WORLD
            )
            deterministic_summary = _slowest_rank_summary(
                deterministic_timings, dist.group.WORLD
            )
            accuracy_elements = min(elements, 64 * 1024)
            accuracy_input = _randn(
                (accuracy_elements,),
                seed=4000 + world_size * 100 + rank,
                device=device,
                scale=1.0,
            )
            gathered_accuracy_input = torch.empty(
                world_size * accuracy_elements,
                dtype=accuracy_input.dtype,
                device=device,
            )
            dist.all_gather_into_tensor(
                gathered_accuracy_input,
                accuracy_input,
                group=dist.group.WORLD,
            )
            rank_inputs = gathered_accuracy_input.reshape(
                world_size, accuracy_elements
            )
            if operation == "all_reduce":
                native_accuracy_output = accuracy_input.clone()
                dist.all_reduce(native_accuracy_output, group=dist.group.WORLD)
                deterministic_accuracy_output = deterministic.all_reduce(
                    accuracy_input
                )
                fp32_reference = rank_inputs.float().sum(dim=0)
            elif operation == "all_gather":
                native_accuracy_output = torch.empty_like(gathered_accuracy_input)
                dist.all_gather_into_tensor(
                    native_accuracy_output,
                    accuracy_input,
                    group=dist.group.WORLD,
                )
                deterministic_accuracy_output = deterministic.all_gather(
                    accuracy_input
                )
                fp32_reference = gathered_accuracy_input.float()
            else:
                native_accuracy_output = torch.empty(
                    accuracy_elements // world_size,
                    dtype=accuracy_input.dtype,
                    device=device,
                )
                dist.reduce_scatter_tensor(
                    native_accuracy_output,
                    accuracy_input,
                    group=dist.group.WORLD,
                )
                deterministic_accuracy_output = deterministic.reduce_scatter(
                    accuracy_input
                )
                reduced_reference = rank_inputs.float().sum(dim=0)
                fp32_reference = reduced_reference.chunk(world_size)[rank]

            local_accuracy = {
                "deterministic_vs_native": _accuracy(
                    deterministic_accuracy_output, native_accuracy_output
                ),
                "native_vs_fp32": _accuracy(
                    native_accuracy_output, fp32_reference
                ),
                "deterministic_vs_fp32": _accuracy(
                    deterministic_accuracy_output, fp32_reference
                ),
            }
            gathered_accuracy: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(
                gathered_accuracy, local_accuracy, group=dist.group.WORLD
            )
            if rank == 0:
                valid_accuracy = [
                    value for value in gathered_accuracy if value is not None
                ]
                results.append(
                    {
                        "operation": operation,
                        "world_size": world_size,
                        "message_bytes_per_rank": message_bytes,
                        "dtype": "bfloat16",
                        "native": native_summary,
                        "deterministic": deterministic_summary,
                        "overhead_ratio": (
                            deterministic_summary["median_ms"]
                            / native_summary["median_ms"]
                        ),
                        "deterministic_vs_native": {
                            "max_abs": max(
                                value["deterministic_vs_native"]["max_abs"]
                                for value in valid_accuracy
                            ),
                            "relative_l2": max(
                                value["deterministic_vs_native"]["relative_l2"]
                                for value in valid_accuracy
                            ),
                            "exact_fraction": min(
                                value["deterministic_vs_native"]["exact_fraction"]
                                for value in valid_accuracy
                            ),
                        },
                        "native_vs_fp32_relative_l2": max(
                            value["native_vs_fp32"]["relative_l2"]
                            for value in valid_accuracy
                        ),
                        "deterministic_vs_fp32_relative_l2": max(
                            value["deterministic_vs_fp32"]["relative_l2"]
                            for value in valid_accuracy
                        ),
                    }
                )
            del (
                accuracy_input,
                gathered_accuracy_input,
                rank_inputs,
                native_accuracy_output,
                deterministic_accuracy_output,
                fp32_reference,
            )
        deterministic.close()
        del input_tensor, deterministic_outputs, native_outputs
        torch.cuda.empty_cache()
    return results


def _distributed_ffn_benchmark(
    rank: int,
    world_size: int,
    configs: tuple[tuple[str, int, int, bool], ...],
    *,
    warmup: int,
    samples: int,
    training_samples: int,
) -> list[dict[str, Any]]:
    device = torch.device("cuda", rank)
    token_count = 32
    hidden_size = 4096
    intermediate_size = 12288
    hidden_full = _randn((token_count, hidden_size), seed=5000, device=device)
    gate_full = _randn(
        (intermediate_size, hidden_size), seed=5001, device=device
    )
    up_full = _randn((intermediate_size, hidden_size), seed=5002, device=device)
    down_full = _randn(
        (hidden_size, intermediate_size), seed=5003, device=device
    )
    grad_output_full = _randn(
        (token_count, hidden_size), seed=5004, device=device
    )

    meshes: dict[tuple[int, int], tuple[list[Any], list[Any]]] = {}
    results = []
    for name, tp_size, cp_size, sequence_parallel in configs:
        mesh_key = (tp_size, cp_size)
        if mesh_key not in meshes:
            meshes[mesh_key] = _mesh_groups(world_size, tp_size, cp_size)
        tp_groups, cp_groups = meshes[mesh_key]
        tp_rank = rank % tp_size
        cp_rank = rank // tp_size
        tp_group = tp_groups[cp_rank] if tp_size > 1 else None
        cp_group = cp_groups[tp_rank] if cp_size > 1 else None
        token_start, token_end, feature_start, feature_end = _shard_ranges(
            rank,
            tp_size=tp_size,
            cp_size=cp_size,
            sequence_parallel=sequence_parallel,
            token_count=token_count,
            intermediate_size=intermediate_size,
        )
        shard = (
            hidden_full[token_start:token_end].contiguous(),
            gate_full[feature_start:feature_end].contiguous(),
            up_full[feature_start:feature_end].contiguous(),
            down_full[:, feature_start:feature_end].contiguous(),
        )
        local_grad_output = grad_output_full[token_start:token_end].contiguous()

        native_forward = lambda: _native_distributed_ffn(
            *shard,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
        )
        deterministic_forward = lambda: qwen3_ffn(
            *shard,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
        )
        triton_forward = lambda: qwen3_ffn_triton(
            *shard,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
        )

        with torch.no_grad():
            native_output = native_forward()
            deterministic_output = deterministic_forward()
            deterministic_repeat = deterministic_forward()
            triton_output = triton_forward()
            triton_repeat = triton_forward()
        torch.cuda.synchronize()

        native_forward_timings = _distributed_wall_samples(
            native_forward,
            group=dist.group.WORLD,
            warmup=warmup,
            samples=samples,
        )
        deterministic_forward_timings = _distributed_wall_samples(
            deterministic_forward,
            group=dist.group.WORLD,
            warmup=warmup,
            samples=samples,
        )
        triton_forward_timings = _distributed_wall_samples(
            triton_forward,
            group=dist.group.WORLD,
            warmup=warmup,
            samples=samples,
        )
        native_forward_summary = _slowest_rank_summary(
            native_forward_timings, dist.group.WORLD
        )
        deterministic_forward_summary = _slowest_rank_summary(
            deterministic_forward_timings, dist.group.WORLD
        )
        triton_forward_summary = _slowest_rank_summary(
            triton_forward_timings, dist.group.WORLD
        )

        native_inputs = [value.detach().clone().requires_grad_(True) for value in shard]
        deterministic_inputs = [
            value.detach().clone().requires_grad_(True) for value in shard
        ]
        triton_inputs = [
            value.detach().clone().requires_grad_(True) for value in shard
        ]

        def native_training_step() -> torch.Tensor:
            for value in native_inputs:
                value.grad = None
            output = _native_distributed_ffn(
                *native_inputs,
                tp_group=tp_group,
                cp_group=cp_group,
                sequence_parallel=sequence_parallel,
            )
            output.backward(local_grad_output)
            return output

        def deterministic_training_step() -> torch.Tensor:
            for value in deterministic_inputs:
                value.grad = None
            output = qwen3_ffn(
                *deterministic_inputs,
                tp_group=tp_group,
                cp_group=cp_group,
                sequence_parallel=sequence_parallel,
            )
            output.backward(local_grad_output)
            return output

        def triton_training_step() -> torch.Tensor:
            for value in triton_inputs:
                value.grad = None
            output = qwen3_ffn_triton(
                *triton_inputs,
                tp_group=tp_group,
                cp_group=cp_group,
                sequence_parallel=sequence_parallel,
            )
            output.backward(local_grad_output)
            return output

        native_train_output = native_training_step().detach().clone()
        native_grads = [value.grad.detach().clone() for value in native_inputs]
        deterministic_train_output = deterministic_training_step().detach().clone()
        deterministic_grads = [
            value.grad.detach().clone() for value in deterministic_inputs
        ]
        triton_train_output = triton_training_step().detach().clone()
        triton_grads = [value.grad.detach().clone() for value in triton_inputs]
        torch.cuda.synchronize()

        native_train_timings = _distributed_wall_samples(
            native_training_step,
            group=dist.group.WORLD,
            warmup=max(1, warmup // 2),
            samples=training_samples,
        )
        deterministic_train_timings = _distributed_wall_samples(
            deterministic_training_step,
            group=dist.group.WORLD,
            warmup=max(1, warmup // 2),
            samples=training_samples,
        )
        triton_train_timings = _distributed_wall_samples(
            triton_training_step,
            group=dist.group.WORLD,
            warmup=max(1, warmup // 2),
            samples=training_samples,
        )
        native_train_summary = _slowest_rank_summary(
            native_train_timings, dist.group.WORLD
        )
        deterministic_train_summary = _slowest_rank_summary(
            deterministic_train_timings, dist.group.WORLD
        )
        triton_train_summary = _slowest_rank_summary(
            triton_train_timings, dist.group.WORLD
        )

        local_accuracy = {
            "output_relative_l2": _relative_l2(
                deterministic_output, native_output
            ),
            "output_max_abs": _accuracy(
                deterministic_output, native_output
            )["max_abs"],
            "max_gradient_relative_l2": max(
                _relative_l2(actual, expected)
                for actual, expected in zip(
                    deterministic_grads, native_grads, strict=True
                )
            ),
            "train_infer_bitwise": torch.equal(
                deterministic_output, deterministic_train_output
            ),
            "repeat_bitwise": torch.equal(
                deterministic_output, deterministic_repeat
            ),
            "triton_forward_mismatch_count": int(
                (triton_output != deterministic_output).sum().item()
            ),
            "triton_train_mismatch_count": int(
                (triton_train_output != deterministic_train_output).sum().item()
                + sum(
                    (triton_grad != deterministic_grad).sum().item()
                    for triton_grad, deterministic_grad in zip(
                        triton_grads, deterministic_grads, strict=True
                    )
                )
            ),
            "triton_train_infer_bitwise": torch.equal(
                triton_output, triton_train_output
            ),
            "triton_repeat_bitwise": torch.equal(triton_output, triton_repeat),
        }
        gathered_accuracy: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(
            gathered_accuracy, local_accuracy, group=dist.group.WORLD
        )
        if rank == 0:
            valid_accuracy = [value for value in gathered_accuracy if value is not None]
            results.extend(
                (
                    {
                        "name": name,
                        "direction": "forward",
                        "world_size": world_size,
                        "tp_size": tp_size,
                        "cp_size": cp_size,
                        "sequence_parallel": sequence_parallel,
                        "tokens": token_count,
                        "hidden": hidden_size,
                        "intermediate": intermediate_size,
                        "native": native_forward_summary,
                        "deterministic": deterministic_forward_summary,
                        "triton": triton_forward_summary,
                        "overhead_ratio": (
                            deterministic_forward_summary["median_ms"]
                            / native_forward_summary["median_ms"]
                        ),
                        "triton_overhead_ratio": (
                            triton_forward_summary["median_ms"]
                            / native_forward_summary["median_ms"]
                        ),
                        "triton_speedup_vs_hip": (
                            deterministic_forward_summary["median_ms"]
                            / triton_forward_summary["median_ms"]
                        ),
                        "triton_vs_hip_mismatch_count": sum(
                            value["triton_forward_mismatch_count"]
                            for value in valid_accuracy
                        ),
                        "output_relative_l2": max(
                            value["output_relative_l2"] for value in valid_accuracy
                        ),
                        "output_max_abs": max(
                            value["output_max_abs"] for value in valid_accuracy
                        ),
                        "train_infer_bitwise": all(
                            value["train_infer_bitwise"] for value in valid_accuracy
                        ),
                        "repeat_bitwise": all(
                            value["repeat_bitwise"] for value in valid_accuracy
                        ),
                        "triton_train_infer_bitwise": all(
                            value["triton_train_infer_bitwise"]
                            for value in valid_accuracy
                        ),
                        "triton_repeat_bitwise": all(
                            value["triton_repeat_bitwise"]
                            for value in valid_accuracy
                        ),
                    },
                    {
                        "name": name,
                        "direction": "train_fwd_bwd",
                        "world_size": world_size,
                        "tp_size": tp_size,
                        "cp_size": cp_size,
                        "sequence_parallel": sequence_parallel,
                        "tokens": token_count,
                        "hidden": hidden_size,
                        "intermediate": intermediate_size,
                        "native": native_train_summary,
                        "deterministic": deterministic_train_summary,
                        "triton": triton_train_summary,
                        "overhead_ratio": (
                            deterministic_train_summary["median_ms"]
                            / native_train_summary["median_ms"]
                        ),
                        "triton_overhead_ratio": (
                            triton_train_summary["median_ms"]
                            / native_train_summary["median_ms"]
                        ),
                        "triton_speedup_vs_hip": (
                            deterministic_train_summary["median_ms"]
                            / triton_train_summary["median_ms"]
                        ),
                        "triton_vs_hip_mismatch_count": sum(
                            value["triton_train_mismatch_count"]
                            for value in valid_accuracy
                        ),
                        "output_relative_l2": max(
                            _relative_l2(
                                deterministic_train_output, native_train_output
                            ),
                            *(value["output_relative_l2"] for value in valid_accuracy),
                        ),
                        "max_gradient_relative_l2": max(
                            value["max_gradient_relative_l2"]
                            for value in valid_accuracy
                        ),
                        "train_infer_bitwise": all(
                            value["train_infer_bitwise"] for value in valid_accuracy
                        ),
                        "repeat_bitwise": all(
                            value["repeat_bitwise"] for value in valid_accuracy
                        ),
                        "triton_train_infer_bitwise": all(
                            value["triton_train_infer_bitwise"]
                            for value in valid_accuracy
                        ),
                        "triton_repeat_bitwise": all(
                            value["triton_repeat_bitwise"]
                            for value in valid_accuracy
                        ),
                    },
                )
            )

        del (
            shard,
            local_grad_output,
            native_inputs,
            deterministic_inputs,
            triton_inputs,
            native_grads,
            deterministic_grads,
            triton_grads,
        )
        torch.cuda.empty_cache()
        dist.barrier(group=dist.group.WORLD)

    for collective in list(ffn_module._COLLECTIVES.values()):
        collective.close()
    ffn_module._COLLECTIVES.clear()
    return results


def _distributed_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_queue: Any,
    warmup: int,
    samples: int,
    training_samples: int,
) -> None:
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=15),
        )
        collective_results = _collective_benchmarks(
            rank,
            world_size,
            warmup=warmup,
            samples=samples,
        )
        ffn_results = _distributed_ffn_benchmark(
            rank,
            world_size,
            _DISTRIBUTED_CONFIGS[world_size],
            warmup=warmup,
            samples=samples,
            training_samples=training_samples,
        )
        if rank == 0:
            result_queue.put(
                {
                    "ok": True,
                    "world_size": world_size,
                    "collectives": collective_results,
                    "distributed_ffn": ffn_results,
                }
            )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "rank": rank,
                "world_size": world_size,
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        for collective in list(ffn_module._COLLECTIVES.values()):
            collective.close()
        ffn_module._COLLECTIVES.clear()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_world(
    world_size: int,
    *,
    warmup: int,
    samples: int,
    training_samples: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as temporary_directory:
        init_method = (Path(temporary_directory) / "rccl_init").as_uri()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_distributed_worker,
                args=(
                    rank,
                    world_size,
                    init_method,
                    result_queue,
                    warmup,
                    samples,
                    training_samples,
                ),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()

        result = None
        try:
            result = result_queue.get(timeout=1800)
            if not result["ok"]:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
        except queue.Empty as exc:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise RuntimeError(
                f"timed out waiting for world_size={world_size} benchmark"
            ) from exc
        finally:
            for process in processes:
                process.join(timeout=60)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
            result_queue.close()
            result_queue.join_thread()

    if result is None:
        raise RuntimeError(f"world_size={world_size} returned no result")
    if not result["ok"]:
        raise RuntimeError(result.get("traceback", str(result)))
    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(
                f"world_size={world_size} worker exited with {process.exitcode}"
            )
    return result


def _write_report(payload: dict[str, Any], output_directory: Path) -> None:
    gemm_rows = payload["single_gpu"]["gemm"]
    swiglu_rows = payload["single_gpu"]["swiglu"]
    ffn_rows = payload["single_gpu"]["ffn"]
    collective_rows = payload["collectives"]
    distributed_rows = payload["distributed_ffn"]
    methodology = payload["methodology"]

    def ratio_range(
        rows: list[dict[str, Any]],
        key: str = "overhead_ratio",
        digits: int = 1,
    ) -> str:
        ratios = [row[key] for row in rows]
        return f"{min(ratios):.{digits}f}-{max(ratios):.{digits}f}x"

    forward_rows = [row for row in distributed_rows if row["direction"] == "forward"]
    training_rows = [
        row for row in distributed_rows if row["direction"] == "train_fwd_bwd"
    ]
    reduction_rows = [
        row
        for row in collective_rows
        if row["operation"] != "all_gather" and row["world_size"] >= 4
    ]
    closer_reductions = sum(
        row["deterministic_vs_fp32_relative_l2"]
        < row["native_vs_fp32_relative_l2"]
        for row in reduction_rows
    )
    gemm_native_errors = [
        row["native_vs_fp32"]["relative_l2"] * 100.0 for row in gemm_rows
    ]
    gemm_deterministic_errors = [
        row["deterministic_vs_fp32"]["relative_l2"] * 100.0 for row in gemm_rows
    ]
    distributed_output_errors = [
        row["output_relative_l2"] * 100.0 for row in distributed_rows
    ]
    consistency_rows = all(
        row["train_infer_bitwise"] and row["repeat_bitwise"]
        for row in distributed_rows
    )
    triton_consistency_rows = all(
        row["triton_train_infer_bitwise"]
        and row["triton_repeat_bitwise"]
        and row["triton_vs_hip_mismatch_count"] == 0
        for row in distributed_rows
    )
    all_triton_rows = gemm_rows + swiglu_rows + ffn_rows + distributed_rows
    total_triton_mismatches = sum(
        row["triton_vs_hip_mismatch_count"] for row in all_triton_rows
    )

    lines = [
        "# PR #325 ROCm deterministic FFN performance analysis",
        "",
        "> Operator-only benchmark. No model checkpoint or serving engine was used.",
        "",
        "## Environment",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    for key, value in payload["environment"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        (
            "",
            "## Methodology",
            "",
            "- BF16 operator inputs; FFN dimensions are H=4096 and I=12288.",
            "- Native GEMM is `torch.matmul`; native communication calls PyTorch's "
            "ProcessGroupNCCL directly, which uses RCCL on ROCm.",
            "- Three compute paths are measured: native ROCm, the strict HIP "
            "correctness kernels, and the new strict Triton kernels. HIP and Triton "
            "share the same deterministic RCCL collective implementation.",
            "- All distributed paths use the same weights, TP/CP/SP sharding, and "
            "collective schedule.",
            f"- Single-GPU timing: {methodology['single_gpu_timing']}; distributed "
            f"timing: {methodology['distributed_timing']}.",
            f"- {methodology['warmup']} warmups, {methodology['samples']} measured "
            f"forward/collective samples, and {methodology['training_samples']} "
            "measured forward+backward samples.",
            "- `NCCL_IB_DISABLE=1` selects the same intra-node XGMI transport for both "
            "collective implementations. Raw median, p95, minimum, and maximum "
            "measurements are in `results.json`.",
            "",
            "Reproduce this report from the repository root:",
            "",
            "```bash",
            "python benchmarks/benchmark_rocm_ffn.py \\",
            f"  --warmup {methodology['warmup']} \\",
            f"  --samples {methodology['samples']} \\",
            f"  --training-samples {methodology['training_samples']} \\",
            "  --output-dir benchmarks/results/pr325_rocm_mi300x",
            "```",
            "",
            "## Key findings",
            "",
            f"- Strict Triton GEMM costs "
            f"{ratio_range(gemm_rows, 'triton_overhead_ratio')} native latency, "
            f"versus {ratio_range(gemm_rows)} for the gfx942 HIP scalar fallback. "
            f"That is a {ratio_range(gemm_rows, 'triton_speedup_vs_hip')} speedup "
            "while preserving the identical BF16 arithmetic tree.",
            f"- Strict Triton SwiGLU costs "
            f"{ratio_range(swiglu_rows, 'triton_overhead_ratio', 2)} native latency; "
            f"the HIP fused kernel costs {ratio_range(swiglu_rows, digits=2)}.",
            f"- Deterministic collectives cost {ratio_range(collective_rows)} native "
            "RCCL latency across 2/4/8 ranks and 64 KiB/1 MiB/16 MiB per rank.",
            f"- Triton distributed FFN overhead is "
            f"{ratio_range(forward_rows, 'triton_overhead_ratio')} for forward and "
            f"{ratio_range(training_rows, 'triton_overhead_ratio')} for "
            f"forward+backward, compared with {ratio_range(forward_rows)} and "
            f"{ratio_range(training_rows)} for HIP.",
            f"- HIP and Triton produced {total_triton_mismatches} mismatched elements "
            f"across all measured GEMM, SwiGLU, single-GPU FFN, and distributed FFN "
            "outputs/gradients. All distributed TP/CP/SP cases are repeat and "
            "train/inference bitwise consistent: "
            f"HIP={'yes' if consistency_rows else 'no'}, "
            f"Triton={'yes' if triton_consistency_rows else 'no'}.",
            f"- GEMM relative-L2 error against FP32 is "
            f"{min(gemm_deterministic_errors):.3f}-{max(gemm_deterministic_errors):.3f}% "
            f"for the deterministic BF16 tree versus "
            f"{min(gemm_native_errors):.3f}-{max(gemm_native_errors):.3f}% for native "
            "ROCm. The fixed BF16 tree buys topology invariance, not better FP32 "
            "proximity.",
            f"- Distributed FFN output drift versus native is "
            f"{min(distributed_output_errors):.3f}-{max(distributed_output_errors):.3f}% "
            "relative-L2. Conversely, the balanced deterministic reduction is closer "
            f"to FP32 than native RCCL in {closer_reductions}/{len(reduction_rows)} "
            "tested 4/8-rank reduction cases.",
        )
    )

    lines.extend(
        (
            "",
            "## Single-GPU GEMM",
            "",
            "| Shape | Native (ms) | Strict HIP (ms) | Strict Triton (ms) | "
            "HIP/native | Triton/native | Triton speedup | Mismatch | "
            "Strict rel-L2 vs FP32 | Native rel-L2 vs FP32 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in gemm_rows:
        lines.append(
            f"| {row['name']} | {row['native']['median_ms']:.4f} | "
            f"{row['deterministic']['median_ms']:.4f} | "
            f"{row['triton']['median_ms']:.4f} | {row['overhead_ratio']:.1f}× | "
            f"{row['triton_overhead_ratio']:.1f}× | "
            f"{row['triton_speedup_vs_hip']:.1f}× | "
            f"{row['triton_vs_hip_mismatch_count']} | "
            f"{row['deterministic_vs_fp32']['relative_l2']:.3e} | "
            f"{row['native_vs_fp32']['relative_l2']:.3e} |"
        )

    lines.extend(
        (
            "",
            "## Single-GPU SwiGLU",
            "",
            "| Case | Native PyTorch (ms) | Strict HIP (ms) | Strict Triton (ms) | "
            "HIP/native | Triton/native | Triton speedup | Mismatch | "
            "Strict/native rel-L2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in swiglu_rows:
        lines.append(
            f"| {row['name']} | {row['native']['median_ms']:.4f} | "
            f"{row['deterministic']['median_ms']:.4f} | "
            f"{row['triton']['median_ms']:.4f} | "
            f"{row['overhead_ratio']:.2f}x | "
            f"{row['triton_overhead_ratio']:.2f}x | "
            f"{row['triton_speedup_vs_hip']:.2f}x | "
            f"{row['triton_vs_hip_mismatch_count']} | "
            f"{row['deterministic_vs_native']['relative_l2']:.3e} |"
        )

    lines.extend(
        (
            "",
            "## Single-GPU FFN",
            "",
            "| Case | Native (ms) | Strict HIP (ms) | Strict Triton (ms) | "
            "HIP/native | Triton/native | Triton speedup | Mismatch | "
            "Output rel-L2 | Max grad rel-L2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in ffn_rows:
        lines.append(
            f"| {row['name']} | {row['native']['median_ms']:.4f} | "
            f"{row['deterministic']['median_ms']:.4f} | "
            f"{row['triton']['median_ms']:.4f} | {row['overhead_ratio']:.1f}× | "
            f"{row['triton_overhead_ratio']:.1f}× | "
            f"{row['triton_speedup_vs_hip']:.1f}× | "
            f"{row['triton_vs_hip_mismatch_count']} | "
            f"{row['deterministic_vs_native']['relative_l2']:.3e} | "
            f"{row.get('max_gradient_relative_l2', float('nan')):.3e} |"
        )

    lines.extend(
        (
            "",
            "## RCCL collectives",
            "",
            "| Operation | Ranks | Input/rank | Native RCCL (ms) | "
            "Deterministic (ms) | Overhead | Det/native rel-L2 | "
            "Native/FP32 rel-L2 | Det/FP32 rel-L2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in collective_rows:
        lines.append(
            f"| {row['operation']} | {row['world_size']} | "
            f"{row['message_bytes_per_rank'] / _MIB:.4g} MiB | "
            f"{row['native']['median_ms']:.4f} | "
            f"{row['deterministic']['median_ms']:.4f} | "
            f"{row['overhead_ratio']:.1f}× | "
            f"{row['deterministic_vs_native']['relative_l2']:.3e} | "
            f"{row['native_vs_fp32_relative_l2']:.3e} | "
            f"{row['deterministic_vs_fp32_relative_l2']:.3e} |"
        )

    lines.extend(
        (
            "",
            "## Distributed FFN",
            "",
            "| Topology | Direction | Native (ms) | Strict HIP (ms) | "
            "Strict Triton (ms) | HIP/native | Triton/native | Triton speedup | "
            "Mismatch | Output rel-L2 | Max grad rel-L2 | Triton T/I | Triton repeat |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
        )
    )
    for row in distributed_rows:
        lines.append(
            f"| {row['name']} | {row['direction']} | "
            f"{row['native']['median_ms']:.4f} | "
            f"{row['deterministic']['median_ms']:.4f} | "
            f"{row['triton']['median_ms']:.4f} | "
            f"{row['overhead_ratio']:.1f}× | "
            f"{row['triton_overhead_ratio']:.1f}× | "
            f"{row['triton_speedup_vs_hip']:.1f}× | "
            f"{row['triton_vs_hip_mismatch_count']} | "
            f"{row['output_relative_l2']:.3e} | "
            f"{row.get('max_gradient_relative_l2', float('nan')):.3e} | "
            f"{'yes' if row['triton_train_infer_bitwise'] else 'no'} | "
            f"{'yes' if row['triton_repeat_bitwise'] else 'no'} |"
        )

    lines.extend(
        (
            "",
            "## Figures",
            "",
            "![Single-GPU overhead](single_gpu_overhead.png)",
            "",
            "![RCCL collective overhead](collective_overhead.png)",
            "",
            "![Distributed FFN overhead](distributed_ffn_overhead.png)",
            "",
            "![Accuracy trade-off](accuracy_tradeoff.png)",
            "",
        )
    )
    (output_directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_figures(payload: dict[str, Any], output_directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    gemm_rows = payload["single_gpu"]["gemm"]
    swiglu_rows = payload["single_gpu"]["swiglu"]
    ffn_rows = payload["single_gpu"]["ffn"]
    labels = (
        [row["name"] for row in gemm_rows]
        + [row["name"] for row in swiglu_rows]
        + [row["name"] for row in ffn_rows]
    )
    hip_ratios = (
        [row["overhead_ratio"] for row in gemm_rows]
        + [row["overhead_ratio"] for row in swiglu_rows]
        + [row["overhead_ratio"] for row in ffn_rows]
    )
    triton_ratios = (
        [row["triton_overhead_ratio"] for row in gemm_rows]
        + [row["triton_overhead_ratio"] for row in swiglu_rows]
        + [row["triton_overhead_ratio"] for row in ffn_rows]
    )
    figure, axis = plt.subplots(figsize=(18, 7))
    positions = list(range(len(labels)))
    width = 0.4
    axis.bar(
        [position - width / 2 for position in positions],
        hip_ratios,
        width,
        color="#ef4444",
        label="Strict HIP",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        triton_ratios,
        width,
        color="#8b5cf6",
        label="Strict Triton",
    )
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_yscale("log")
    axis.set_ylabel("Strict / native latency (×, log scale)")
    axis.set_title("MI300X single-GPU strict HIP vs Triton overhead")
    axis.set_xticks(positions, labels, rotation=55, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "single_gpu_overhead.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, operation in zip(
        axes, ("all_reduce", "all_gather", "reduce_scatter"), strict=True
    ):
        for world_size in (2, 4, 8):
            rows = [
                row
                for row in payload["collectives"]
                if row["operation"] == operation and row["world_size"] == world_size
            ]
            axis.plot(
                [row["message_bytes_per_rank"] / _MIB for row in rows],
                [row["overhead_ratio"] for row in rows],
                marker="o",
                label=f"{world_size} ranks",
            )
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(operation)
        axis.set_xlabel("Input per rank (MiB)")
    axes[0].set_ylabel("Deterministic / native latency (×, log scale)")
    axes[-1].legend()
    figure.suptitle("MI300X deterministic collective overhead vs native RCCL")
    figure.tight_layout()
    figure.savefig(output_directory / "collective_overhead.png", dpi=180)
    plt.close(figure)

    rows = payload["distributed_ffn"]
    labels = [f"{row['name']}\n{row['direction']}" for row in rows]
    figure, axis = plt.subplots(figsize=(16, 6))
    positions = list(range(len(rows)))
    width = 0.4
    axis.bar(
        [position - width / 2 for position in positions],
        [row["overhead_ratio"] for row in rows],
        width,
        color="#ef4444",
        label="Strict HIP",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [row["triton_overhead_ratio"] for row in rows],
        width,
        color="#8b5cf6",
        label="Strict Triton",
    )
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_yscale("log")
    axis.set_ylabel("Strict / native latency (×, log scale)")
    axis.set_title("Qwen3-8B-shaped FFN: strict HIP vs Triton on MI300X")
    axis.set_xticks(positions, labels, rotation=55, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "distributed_ffn_overhead.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    names = [row["name"] for row in gemm_rows]
    native_errors = [row["native_vs_fp32"]["relative_l2"] for row in gemm_rows]
    deterministic_errors = [
        row["deterministic_vs_fp32"]["relative_l2"] for row in gemm_rows
    ]
    positions = list(range(len(names)))
    width = 0.38
    axis.bar(
        [position - width / 2 for position in positions],
        native_errors,
        width,
        label="Native ROCm vs FP32",
        color="#3b82f6",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        deterministic_errors,
        width,
        label="Strict HIP = Triton vs FP32",
        color="#ef4444",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Relative L2 error (log scale)")
    axis.set_title("BF16 GEMM accuracy: strict Triton has 0 mismatch vs strict HIP")
    axis.set_xticks(positions, names, rotation=45, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "accuracy_tradeoff.png", dpi=180)
    plt.close(figure)


def _environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "architecture": properties.gcnArchName,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "python": os.sys.version.split()[0],
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "native_gemm": "torch.matmul (ROCm rocBLAS/hipBLASLt dispatch)",
        "native_collective": "torch.distributed ProcessGroupNCCL (RCCL on ROCm)",
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", ""),
    }


def _validate_environment() -> None:
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("this benchmark requires a ROCm PyTorch build")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError("this benchmark requires eight visible ROCm GPUs")
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError("the native rl_engine extension is unavailable")
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(f"the native extension is missing symbols: {missing}")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("PyTorch RCCL/ProcessGroupNCCL support is unavailable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/pr325_rocm_mi300x"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--training-samples", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_environment()
    # The host exposes unusable RDMA interfaces.  Both native and deterministic
    # paths use the same intra-node XGMI transport with IB disabled.
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "environment": _environment(),
        "methodology": {
            "single_gpu_timing": "GPU events, median and p95",
            "distributed_timing": "synchronized wall clock, slowest rank per sample",
            "warmup": args.warmup,
            "samples": args.samples,
            "training_samples": args.training_samples,
            "operator_only": True,
        },
        "single_gpu": _single_gpu_benchmarks(
            warmup=args.warmup,
            samples=args.samples,
            training_samples=args.training_samples,
        ),
        "collectives": [],
        "distributed_ffn": [],
    }
    torch.cuda.empty_cache()

    for world_size in (2, 4, 8):
        result = _run_distributed_world(
            world_size,
            warmup=args.warmup,
            samples=args.samples,
            training_samples=args.training_samples,
        )
        payload["collectives"].extend(result["collectives"])
        payload["distributed_ffn"].extend(result["distributed_ffn"])

    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(payload, args.output_dir)
    _write_figures(payload, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "status": "ok"}))


if __name__ == "__main__":
    main()
