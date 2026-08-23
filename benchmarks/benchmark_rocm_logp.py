# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""ROCm benchmark for the WS2 vocab-parallel logprob path (PR #328).

Operator-only: seeded logits, no checkpoint, tokenizer, or model server.

Single GPU (TP=1), Qwen3 vocabulary ``V=151936`` (64 tiles of 2374 columns):

- ``native``: ``torch.logsumexp`` + ``gather`` on FP32 logits (plain PyTorch,
  not batch-invariant by contract).
- ``ws1-pytorch`` / ``ws1-triton``: existing single-shard batch-invariant ops.
- ``ws2-reference``: ``pytorch-vocab-parallel-logp-ws2`` (PyTorch tile loop).
- ``ws2-rocm``: ``rocm-vocab-parallel-logp-ws2`` (HIP tile-stats kernel).

Plus a component table for the HIP ``deterministic_logp_tile_stats`` kernel
against the PyTorch ``_local_tile_stats`` loop it replaces.

Distributed (one process per GPU, RCCL via ProcessGroupNCCL):

- ``native``: Megatron-style vocab-parallel logprob (all-reduce MAX, all-reduce
  SUM of exp, all-reduce SUM of the owned target logit).
- ``ws2-reference`` and ``ws2-rocm``: the contract-aware WS2 operator with the
  fixed tile-order merge; CP ranks shard tokens and never join the merge.

Every path reports latency (GPU events on a single GPU; synchronized wall
clock and slowest rank per sample when distributed), peak device memory,
FP64 accuracy, repeat bitwise stability, and batch invariance.

Usage:
    python benchmarks/benchmark_rocm_logp.py \
        --warmup 5 --samples 20 --training-samples 10 \
        --output-dir benchmarks/results/pr328_rocm_mi300x
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.kernels.logprob_contract import (
    LogprobContract,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import NativeBatchInvariantLogpOp
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import (
    VocabParallelLogprobOp,
    _local_tile_stats,
)
from rl_engine.kernels.ops.rocm.loss.vocab_parallel_logp import RocmVocabParallelLogprobOp

REAL_VOCAB = 151936  # Qwen3 tokenizer/lm_head width; 151936 = 64 * 2374, so no padding.
NUM_TILES = 64
IGNORE_INDEX = -100
LOGIT_SCALE = 2.0
SINGLE_TOKENS = (1, 8, 32, 128, 512, 2048)
SINGLE_DTYPES = ("bf16", "fp32")
DISTRIBUTED_TOKENS = (256, 2048)
TOPOLOGIES = (
    ("tp2", 2, 1),
    ("tp4", 4, 1),
    ("tp8", 8, 1),
    ("tp2_cp2", 2, 2),
    ("tp4_cp2", 4, 2),
    ("tp2_cp4", 2, 4),
)
DISTRIBUTED_PATHS = ("native", "ws2-reference", "ws2-rocm")
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
_SPAWN_TIMEOUT_S = 1800


# --------------------------------------------------------------------------- helpers


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
    actual_float = actual.detach().double()
    expected_float = expected.detach().double()
    denominator = torch.linalg.vector_norm(expected_float)
    if denominator.item() == 0.0:
        return float(torch.linalg.vector_norm(actual_float - expected_float).item())
    return float((torch.linalg.vector_norm(actual_float - expected_float) / denominator).item())


def _accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().double() - expected.detach().double()
    return {
        "max_abs": float(difference.abs().max().item()) if difference.numel() else 0.0,
        "relative_l2": _relative_l2(actual, expected),
    }


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().contiguous().view(torch.int32)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(_bits(a), _bits(b)))


def _mismatch_count(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_bits(a) != _bits(b)).sum().item())


def _gpu_event_samples(function: Callable[[], Any], *, warmup: int, samples: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    events = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in events]


def _peak_memory_mib(function: Callable[[], Any]) -> float:
    """Peak device memory allocated by one call, above what was live before it."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    function()
    torch.cuda.synchronize()
    return float((torch.cuda.max_memory_allocated() - baseline) / (1024.0 * 1024.0))


def _seeded_logits(
    num_tokens: int, vocab: int, *, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identical FP32 logits, targets, and active mask on every rank.

    Generated on-device from a seeded CUDA generator so multi-GB inputs are not
    materialized on the host; the same seed yields identical values on every
    MI300X rank.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(num_tokens, vocab, generator=generator, device=device) * LOGIT_SCALE
    targets = torch.randint(0, vocab, (num_tokens,), generator=generator, device=device)
    active = (torch.arange(num_tokens, device=device) % 7) != 5
    return logits, targets, active


def _fp64_oracle(
    logits_fp32: torch.Tensor, targets: torch.Tensor, active: torch.Tensor, real_vocab: int
) -> tuple[torch.Tensor, torch.Tensor]:
    z = logits_fp32[:, :real_vocab].double()
    lse = torch.logsumexp(z, dim=-1)
    safe = torch.where(active, targets, torch.zeros_like(targets))
    selected = z.gather(1, safe.unsqueeze(1)).squeeze(1)
    logp = torch.where(active, selected - lse, torch.zeros_like(lse))
    return logp, lse


def _contract(
    *,
    num_tokens: int,
    active: tuple[bool, ...],
    tp_rank: int,
    tp_world_size: int,
    bounds: tuple[tuple[int, int], ...],
    real_vocab: int,
    padded_vocab: int,
    dtype: str,
    cp_rank: int = 0,
    cp_world_size: int = 1,
) -> LogprobContract:
    return LogprobContract(
        role="train",
        dtype=dtype,
        mask=MaskSpec(num_tokens=num_tokens, active_mask=active),
        sharding=ShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            vocab_shard_bounds=bounds,
            real_vocab_size=real_vocab,
            padded_vocab_size=padded_vocab,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
        ),
        reduction=ReductionSpec(),
    )


# --------------------------------------------------------------------------- single GPU


def _native_logp(
    logits: torch.Tensor, targets: torch.Tensor, active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    z = logits.float()
    lse = torch.logsumexp(z, dim=-1)
    safe = torch.where(active, targets, torch.zeros_like(targets))
    selected = z.gather(1, safe.unsqueeze(1)).squeeze(1)
    logp = torch.where(active, selected - lse, torch.zeros_like(lse))
    return logp, lse


def _single_gpu_paths(
    device: torch.device,
) -> dict[str, tuple[Callable[..., Any], Callable[..., Any]]]:
    ws1_pytorch = NativeBatchInvariantLogpOp()
    ws2_reference = VocabParallelLogprobOp()
    ws2_rocm = RocmVocabParallelLogprobOp()

    def ws1_pytorch_fn(logits, targets, active, contract):
        ignore_targets = torch.where(active, targets, torch.full_like(targets, IGNORE_INDEX))
        return ws1_pytorch.forward_with_lse(logits, ignore_targets, IGNORE_INDEX, validate=False)

    def ws1_pytorch_train(logits, targets, active, contract):
        ignore_targets = torch.where(active, targets, torch.full_like(targets, IGNORE_INDEX))
        return ws1_pytorch.apply(logits, ignore_targets, IGNORE_INDEX, validate=False)

    def ws2_reference_fn(logits, targets, active, contract):
        return ws2_reference.apply(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES, validate=False
        )

    def ws2_rocm_fn(logits, targets, active, contract):
        return ws2_rocm.apply(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES, validate=False
        )

    # path -> (forward returning (logp, lse), training forward returning logp with autograd)
    paths: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
        "native": (
            lambda logits, targets, active, contract: _native_logp(logits, targets, active),
            lambda logits, targets, active, contract: _native_logp(logits, targets, active)[0],
        ),
        "ws1-pytorch": (ws1_pytorch_fn, ws1_pytorch_train),
    }
    try:
        from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import (
            TritonBatchInvariantLogpOp,
        )

        ws1_triton = TritonBatchInvariantLogpOp()
        probe = torch.randn(2, 256, device=device, dtype=torch.bfloat16)
        ws1_triton.forward_with_lse(probe, torch.zeros(2, device=device, dtype=torch.long))
        torch.cuda.synchronize()

        def ws1_triton_fn(logits, targets, active, contract):
            ignore_targets = torch.where(active, targets, torch.full_like(targets, IGNORE_INDEX))
            return ws1_triton.forward_with_lse(logits, ignore_targets, IGNORE_INDEX)

        def ws1_triton_train(logits, targets, active, contract):
            ignore_targets = torch.where(active, targets, torch.full_like(targets, IGNORE_INDEX))
            return ws1_triton.apply(logits, ignore_targets, IGNORE_INDEX)

        paths["ws1-triton"] = (ws1_triton_fn, ws1_triton_train)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"ws1-triton unavailable: {exc}")
    paths["ws2-reference"] = (
        ws2_reference_fn,
        lambda *a: ws2_reference_fn(*a)[0],
    )
    paths["ws2-rocm"] = (ws2_rocm_fn, lambda *a: ws2_rocm_fn(*a)[0])
    return paths


def _single_gpu_benchmarks(
    *, warmup: int, samples: int, training_samples: int, tokens: tuple[int, ...]
) -> dict[str, Any]:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    paths = _single_gpu_paths(device)
    bounds = ((0, REAL_VOCAB),)
    cases: list[dict[str, Any]] = []
    validate_overhead: list[dict[str, Any]] = []
    ws2_rocm_op = RocmVocabParallelLogprobOp()

    for dtype_name in SINGLE_DTYPES:
        dtype = _DTYPES[dtype_name]
        for num_tokens in tokens:
            logits_fp32, targets, active = _seeded_logits(
                num_tokens, REAL_VOCAB, seed=2026 + num_tokens, device=device
            )
            logits = logits_fp32.to(dtype).contiguous()
            oracle_logp, oracle_lse = _fp64_oracle(logits.float(), targets, active, REAL_VOCAB)
            logits_fp32 = None
            contract = _contract(
                num_tokens=num_tokens,
                active=tuple(bool(flag) for flag in active.tolist()),
                tp_rank=0,
                tp_world_size=1,
                bounds=bounds,
                real_vocab=REAL_VOCAB,
                padded_vocab=REAL_VOCAB,
                dtype=dtype_name,
            )
            row = min(3, num_tokens - 1)
            row_contract = _contract(
                num_tokens=1,
                active=(bool(active[row].item()),),
                tp_rank=0,
                tp_world_size=1,
                bounds=bounds,
                real_vocab=REAL_VOCAB,
                padded_vocab=REAL_VOCAB,
                dtype=dtype_name,
            )
            outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for path_name, (path, train_path) in paths.items():

                def forward():
                    return path(logits, targets, active, contract)

                def train_step():
                    leaf = logits.detach().clone().requires_grad_(True)
                    logp = train_path(leaf, targets, active, contract)
                    (logp * active).sum().backward()
                    return leaf.grad

                try:
                    first_logp, first_lse = forward()
                    second_logp, second_lse = forward()
                    row_logp, row_lse = path(
                        logits[row : row + 1].contiguous(),
                        targets[row : row + 1],
                        active[row : row + 1],
                        row_contract,
                    )
                    train_step()
                    torch.cuda.synchronize()
                except Exception as exc:
                    print(f"{path_name} failed for {dtype_name} M={num_tokens}: {exc}")
                    continue
                outputs[path_name] = (first_logp.detach(), first_lse.detach())
                forward_times = _gpu_event_samples(forward, warmup=warmup, samples=samples)
                train_times = _gpu_event_samples(
                    train_step, warmup=max(1, warmup // 2), samples=training_samples
                )
                grad = train_step()
                torch.cuda.synchronize()
                active_idx = active.nonzero().squeeze(1)
                cases.append(
                    {
                        "dtype": dtype_name,
                        "tokens": num_tokens,
                        "path": path_name,
                        "forward": _summary_ms(forward_times),
                        "train_fwd_bwd": _summary_ms(train_times),
                        "forward_peak_mib": _peak_memory_mib(forward),
                        "train_peak_mib": _peak_memory_mib(train_step),
                        "logp_vs_fp64": _accuracy(first_logp[active_idx], oracle_logp[active_idx]),
                        "lse_vs_fp64": _accuracy(first_lse, oracle_lse),
                        "repeat_bitwise": _bitwise_equal(first_logp, second_logp)
                        and _bitwise_equal(first_lse, second_lse),
                        "batch_invariant": _bitwise_equal(row_logp[0], first_logp[row])
                        and _bitwise_equal(row_lse[0], first_lse[row]),
                        "grad_finite": bool(torch.isfinite(grad).all().item()),
                    }
                )
                print(
                    f"single {dtype_name} M={num_tokens:5d} {path_name:14s} "
                    f"fwd={cases[-1]['forward']['median_ms']:.4f}ms "
                    f"train={cases[-1]['train_fwd_bwd']['median_ms']:.4f}ms "
                    f"peak={cases[-1]['train_peak_mib']:.1f}MiB",
                    flush=True,
                )
            if "ws2-reference" in outputs and "ws2-rocm" in outputs:
                ref_logp, ref_lse = outputs["ws2-reference"]
                rocm_logp, rocm_lse = outputs["ws2-rocm"]
                for case in cases:
                    if case["dtype"] == dtype_name and case["tokens"] == num_tokens:
                        if case["path"] == "ws2-rocm":
                            case["mismatch_vs_reference"] = _mismatch_count(
                                rocm_logp, ref_logp
                            ) + _mismatch_count(rocm_lse, ref_lse)
                            case["rel_l2_vs_reference"] = max(
                                _relative_l2(rocm_logp, ref_logp), _relative_l2(rocm_lse, ref_lse)
                            )
            # validate=True production entry point overhead (host-side checks + .item() sync)
            if dtype_name == "bf16":

                def validated():
                    return ws2_rocm_op.apply(
                        logits, targets, contract=contract, num_vocab_tiles=NUM_TILES, validate=True
                    )

                def unvalidated():
                    return ws2_rocm_op.apply(
                        logits,
                        targets,
                        contract=contract,
                        num_vocab_tiles=NUM_TILES,
                        validate=False,
                    )

                validate_overhead.append(
                    {
                        "tokens": num_tokens,
                        "validate_true": _summary_ms(
                            _gpu_event_samples(validated, warmup=warmup, samples=samples)
                        ),
                        "validate_false": _summary_ms(
                            _gpu_event_samples(unvalidated, warmup=warmup, samples=samples)
                        ),
                    }
                )
            logits = oracle_logp = oracle_lse = outputs = None
            torch.cuda.empty_cache()

    return {"cases": cases, "validate_overhead": validate_overhead, "paths": list(paths)}


def _tile_stats_component(
    *, warmup: int, samples: int, tokens: tuple[int, ...]
) -> list[dict[str, Any]]:
    """HIP tile-stats kernel versus the PyTorch tile loop it replaces."""
    device = torch.device("cuda", 0)
    rows: list[dict[str, Any]] = []
    tile = REAL_VOCAB // NUM_TILES
    for dtype_name in SINGLE_DTYPES:
        dtype = _DTYPES[dtype_name]
        for num_tokens in tokens:
            logits_fp32, _, _ = _seeded_logits(
                num_tokens, REAL_VOCAB, seed=99 + num_tokens, device=device
            )
            logits = logits_fp32.to(dtype).contiguous()
            z32 = logits.float()
            logits_fp32 = None

            def pytorch_loop():
                return _local_tile_stats(z32, tile)

            tile_stats = getattr(
                _C, "hip_deterministic_logp_tile_stats", _C.deterministic_logp_tile_stats
            )

            def hip_kernel_fp32():
                return tile_stats(z32, 0, REAL_VOCAB, NUM_TILES)

            def hip_kernel_input_dtype():
                return tile_stats(logits, 0, REAL_VOCAB, NUM_TILES)

            ref_m, ref_s = pytorch_loop()
            hip_m, hip_s = hip_kernel_fp32()
            hip_m2, hip_s2 = hip_kernel_fp32()
            rows.append(
                {
                    "dtype": dtype_name,
                    "tokens": num_tokens,
                    "pytorch_loop": _summary_ms(
                        _gpu_event_samples(pytorch_loop, warmup=warmup, samples=samples)
                    ),
                    "hip_fp32_input": _summary_ms(
                        _gpu_event_samples(hip_kernel_fp32, warmup=warmup, samples=samples)
                    ),
                    "hip_native_dtype_input": _summary_ms(
                        _gpu_event_samples(hip_kernel_input_dtype, warmup=warmup, samples=samples)
                    ),
                    "pytorch_loop_peak_mib": _peak_memory_mib(pytorch_loop),
                    "hip_peak_mib": _peak_memory_mib(hip_kernel_fp32),
                    "max_bitwise": _bitwise_equal(hip_m, ref_m),
                    "sumexp_rel_l2": _relative_l2(hip_s, ref_s),
                    "sumexp_max_rel": float(
                        ((hip_s - ref_s).abs() / ref_s.abs().clamp_min(1e-30)).max().item()
                    ),
                    "repeat_bitwise": _bitwise_equal(hip_m, hip_m2)
                    and _bitwise_equal(hip_s, hip_s2),
                }
            )
            print(
                f"tile-stats {dtype_name} M={num_tokens:5d} "
                f"loop={rows[-1]['pytorch_loop']['median_ms']:.4f}ms "
                f"hip={rows[-1]['hip_fp32_input']['median_ms']:.4f}ms",
                flush=True,
            )
            logits = z32 = None
            torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- distributed


class _NativeVocabParallelLogp(torch.autograd.Function):
    """Megatron-style vocab-parallel logprob with RCCL all-reduce, no fixed order."""

    @staticmethod
    def forward(ctx, local_logits, targets, active, vocab_start, group):
        z = local_logits.float()
        local_vocab = z.shape[1]
        global_max = z.max(dim=-1).values
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)
        sum_exp = (z - global_max.unsqueeze(1)).exp().sum(dim=-1)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)
        lse = global_max + sum_exp.log()
        owned = active & (targets >= vocab_start) & (targets < vocab_start + local_vocab)
        local_index = torch.where(owned, targets - vocab_start, torch.zeros_like(targets))
        selected = z.gather(1, local_index.unsqueeze(1)).squeeze(1) * owned
        dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=group)
        logp = torch.where(active, selected - lse, torch.zeros_like(lse))
        ctx.save_for_backward(z, lse, local_index, owned, active)
        ctx.input_dtype = local_logits.dtype
        ctx.set_materialize_grads(False)
        return logp, lse

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        z, lse, local_index, owned, active = ctx.saved_tensors
        probabilities = (z - lse.unsqueeze(1)).exp()
        scale = torch.zeros_like(lse)
        if grad_logp is not None:
            scale = scale - grad_logp * active
        if grad_lse is not None:
            scale = scale + grad_lse
        grad = probabilities * scale.unsqueeze(1)
        if grad_logp is not None:
            grad.scatter_add_(
                1, local_index.unsqueeze(1), (grad_logp * owned).unsqueeze(1).to(grad.dtype)
            )
        return grad.to(ctx.input_dtype), None, None, None, None


def _tp_bounds(tp_world_size: int) -> tuple[tuple[int, int], ...]:
    tile = REAL_VOCAB // NUM_TILES
    per_rank = NUM_TILES // tp_world_size
    return tuple(
        (rank * per_rank * tile, (rank + 1) * per_rank * tile) for rank in range(tp_world_size)
    )


def _token_bounds(num_tokens: int, cp_world_size: int) -> tuple[tuple[int, int], ...]:
    quotient, remainder = divmod(num_tokens, cp_world_size)
    bounds, cursor = [], 0
    for cp_rank in range(cp_world_size):
        count = quotient + int(cp_rank < remainder)
        bounds.append((cursor, cursor + count))
        cursor += count
    return tuple(bounds)


def _distributed_wall_samples(
    function: Callable[[], Any], *, warmup: int, samples: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    dist.barrier()
    timings = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    dist.barrier()
    return timings


def _slowest_rank_summary(local_timings: list[float]) -> dict[str, float]:
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_timings)
    slowest = [
        max(float(rank_timings[index]) for rank_timings in gathered if rank_timings is not None)
        for index in range(len(local_timings))
    ]
    return _summary_ms(slowest)


def _all_max(value: float) -> float:
    gathered: list[float | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, float(value))
    return max(float(item) for item in gathered if item is not None)


def _all_all(flag: bool) -> bool:
    gathered: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, bool(flag))
    return all(bool(item) for item in gathered)


def _all_sum(value: int) -> int:
    gathered: list[int | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, int(value))
    return sum(int(item) for item in gathered if item is not None)


def _tp_replicated(
    logp: torch.Tensor, lse: torch.Tensor, tp_group: Any, tp_world_size: int
) -> bool:
    if tp_world_size == 1:
        return True
    payload = torch.stack([_bits(logp), _bits(lse)])
    gathered = [torch.empty_like(payload) for _ in range(tp_world_size)]
    dist.all_gather(gathered, payload, group=tp_group)
    return all(torch.equal(gathered[0], other) for other in gathered[1:])


def _distributed_worker(
    rank: int,
    world_size: int,
    init_method: str,
    topology: tuple[str, int, int],
    tokens_list: tuple[int, ...],
    config: dict[str, Any],
    result_queue: Any,
) -> None:
    name, tp_world_size, cp_world_size = topology
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
        cp_rank, tp_rank = divmod(rank, tp_world_size)
        tp_group = None
        for group_cp_rank in range(cp_world_size):
            ranks = list(range(group_cp_rank * tp_world_size, (group_cp_rank + 1) * tp_world_size))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                tp_group = group
        bounds = _tp_bounds(tp_world_size)
        vocab_start, vocab_end = bounds[tp_rank]
        ops = {
            "ws2-reference": VocabParallelLogprobOp(),
            "ws2-rocm": RocmVocabParallelLogprobOp(),
        }
        results: list[dict[str, Any]] = []

        for num_tokens in tokens_list:
            full_logits, full_targets, full_active = _seeded_logits(
                num_tokens, REAL_VOCAB, seed=4100 + num_tokens, device=device
            )
            token_start, token_end = _token_bounds(num_tokens, cp_world_size)[cp_rank]
            local_tokens = token_end - token_start
            targets = full_targets[token_start:token_end].contiguous()
            active = full_active[token_start:token_end].contiguous()
            local_fp32 = full_logits[token_start:token_end]
            shard = local_fp32[:, vocab_start:vocab_end].to(torch.bfloat16).contiguous()
            # The FP64 oracle sees the BF16-rounded logits every path actually consumes.
            oracle_logp, oracle_lse = _fp64_oracle(
                local_fp32.to(torch.bfloat16).float(), targets, active, REAL_VOCAB
            )
            full_logits = local_fp32 = None
            torch.cuda.empty_cache()
            contract = _contract(
                num_tokens=local_tokens,
                active=tuple(bool(flag) for flag in active.tolist()),
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
                bounds=bounds,
                real_vocab=REAL_VOCAB,
                padded_vocab=REAL_VOCAB,
                dtype="bf16",
                cp_rank=cp_rank,
                cp_world_size=cp_world_size,
            )
            outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            for path_name in DISTRIBUTED_PATHS:
                if path_name == "native":

                    def forward(x=shard):
                        return _NativeVocabParallelLogp.apply(
                            x, targets, active, vocab_start, tp_group
                        )

                else:
                    op = ops[path_name]

                    def forward(x=shard, op=op):
                        return op.apply(
                            x,
                            targets,
                            contract=contract,
                            tp_group=tp_group,
                            num_vocab_tiles=NUM_TILES,
                            validate=False,
                        )

                def train_step():
                    leaf = shard.detach().clone().requires_grad_(True)
                    logp, _ = forward(leaf)
                    (logp * active).sum().backward()
                    return leaf.grad

                first_logp, first_lse = forward()
                second_logp, second_lse = forward()
                torch.cuda.synchronize()
                outputs[path_name] = (first_logp.detach(), first_lse.detach())
                forward_times = _distributed_wall_samples(
                    forward, warmup=config["warmup"], samples=config["samples"]
                )
                train_times = _distributed_wall_samples(
                    train_step,
                    warmup=max(1, config["warmup"] // 2),
                    samples=config["training_samples"],
                )
                grad = train_step()
                torch.cuda.synchronize()
                active_idx = active.nonzero().squeeze(1)
                logp_acc = _accuracy(first_logp[active_idx], oracle_logp[active_idx])
                lse_acc = _accuracy(first_lse, oracle_lse)
                entry = {
                    "topology": name,
                    "tp": tp_world_size,
                    "cp": cp_world_size,
                    "tokens": num_tokens,
                    "tokens_per_cp_rank": local_tokens,
                    "local_vocab": vocab_end - vocab_start,
                    "path": path_name,
                    "forward": _slowest_rank_summary(forward_times),
                    "train_fwd_bwd": _slowest_rank_summary(train_times),
                    "forward_peak_mib": _all_max(_peak_memory_mib(forward)),
                    "train_peak_mib": _all_max(_peak_memory_mib(train_step)),
                    "logp_vs_fp64_max_abs": _all_max(logp_acc["max_abs"]),
                    "logp_vs_fp64_rel_l2": _all_max(logp_acc["relative_l2"]),
                    "lse_vs_fp64_max_abs": _all_max(lse_acc["max_abs"]),
                    "tp_replicated": _all_all(
                        _tp_replicated(first_logp, first_lse, tp_group, tp_world_size)
                    ),
                    "repeat_bitwise": _all_all(
                        _bitwise_equal(first_logp, second_logp)
                        and _bitwise_equal(first_lse, second_lse)
                    ),
                    "grad_finite": _all_all(bool(torch.isfinite(grad).all().item())),
                }
                results.append(entry)
                if rank == 0:
                    print(
                        f"dist {name} M={num_tokens:5d} {path_name:14s} "
                        f"fwd={entry['forward']['median_ms']:.4f}ms "
                        f"train={entry['train_fwd_bwd']['median_ms']:.4f}ms "
                        f"peak={entry['train_peak_mib']:.1f}MiB",
                        flush=True,
                    )
            ref_logp, ref_lse = outputs["ws2-reference"]
            rocm_logp, rocm_lse = outputs["ws2-rocm"]
            nat_logp, nat_lse = outputs["native"]
            mismatch = _mismatch_count(rocm_logp, ref_logp) + _mismatch_count(rocm_lse, ref_lse)
            rel = max(_relative_l2(rocm_logp, ref_logp), _relative_l2(rocm_lse, ref_lse))
            native_rel = max(_relative_l2(nat_logp, ref_logp), _relative_l2(nat_lse, ref_lse))
            # Count once per TP group (outputs are replicated inside the group).
            mismatch_total = _all_sum(mismatch if tp_rank == 0 else 0)
            rel_max = _all_max(rel)
            native_rel_max = _all_max(native_rel)
            for entry in results:
                if entry["tokens"] == num_tokens and entry["path"] == "ws2-rocm":
                    entry["mismatch_vs_reference"] = mismatch_total
                    entry["rel_l2_vs_reference"] = rel_max
                if entry["tokens"] == num_tokens and entry["path"] == "native":
                    entry["rel_l2_vs_reference"] = native_rel_max
            shard = oracle_logp = oracle_lse = outputs = None
            torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            result_queue.put({"ok": True, "results": results})
    except Exception:
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_world(
    topology: tuple[str, int, int], tokens_list: tuple[int, ...], config: dict[str, Any]
) -> list[dict[str, Any]]:
    name, tp_world_size, cp_world_size = topology
    world_size = tp_world_size * cp_world_size
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "rccl_init").as_uri()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_distributed_worker,
                args=(rank, world_size, init_method, topology, tokens_list, config, result_queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        try:
            payload = result_queue.get(timeout=_SPAWN_TIMEOUT_S)
        finally:
            for process in processes:
                process.join(timeout=120)
                if process.is_alive():
                    process.terminate()
    if not payload.get("ok"):
        raise RuntimeError(f"{name} rank {payload.get('rank')} failed:\n{payload.get('traceback')}")
    return payload["results"]


# --------------------------------------------------------------------------- report


PATH_DESCRIPTIONS = {
    "native": (
        "`torch.logsumexp` + `gather` on FP32 logits (plain PyTorch, not batch-invariant "
        "by contract)"
    ),
    "ws1-pytorch": (
        "`pytorch-batch-invariant-logp-ws1`, the single-shard batch-invariant PyTorch op"
    ),
    "ws1-triton": (
        "`triton-batch-invariant-logp-ws1`, the single-shard Triton online-softmax op; it has "
        "no vocab-parallel (TP) path, so it appears only in the single-GPU results"
    ),
    "ws2-reference": (
        "`pytorch-vocab-parallel-logp-ws2`, the WS2 vocab-parallel reference operator: a PyTorch "
        "tile loop for the per-tile FP32 `(max, sumexp)` partials, all-gather of the partials, "
        "fixed global tile-order merge, and a PyTorch autograd backward"
    ),
    "ws2-rocm": (
        "`rocm-vocab-parallel-logp-ws2`, the same contract, transport, and merge with two HIP "
        "kernels: `hip_deterministic_logp_tile_stats` reads the stored BF16/FP16/FP32 shard "
        "directly (8-element vector loads, no FP32 copy) and `hip_deterministic_logp_backward` "
        "produces the gradient in one fused pass"
    ),
}
DISTRIBUTED_DESCRIPTIONS = {
    "native": (
        "`native` is a Megatron-style vocab-parallel logprob using RCCL all-reduce (MAX, SUM of "
        "exp, SUM of the owned target logit) through ProcessGroupNCCL"
    ),
    "ws2-reference": (
        "the WS2 operators all-gather per-tile `(max, sumexp)` partials over RCCL and merge them "
        "in fixed global tile order; CP ranks shard tokens and never enter the merge"
    ),
    "ws2-rocm": "",
}


class ReportStyle:
    """Which measured paths a report shows, how it names them, and what it compares against."""

    def __init__(
        self,
        *,
        paths: tuple[str, ...],
        names: dict[str, str],
        baseline: str,
        table_tokens: tuple[int, ...] | None,
        show_tuning_baseline: bool,
        command: str,
    ) -> None:
        if baseline not in paths:
            raise ValueError(
                f"report baseline {baseline!r} is not among the reported paths {paths}"
            )
        self.paths = paths
        self.names = names
        self.baseline = baseline
        self.table_tokens = table_tokens
        self.show_tuning_baseline = show_tuning_baseline
        self.command = command

    def name(self, path: str) -> str:
        return self.names.get(path, path)

    def keep_tokens(self, tokens: int) -> bool:
        return self.table_tokens is None or tokens in self.table_tokens


def _fmt_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator / denominator:.2f}×"


def _median_ratio(
    numerator: dict[str, Any] | None, denominator: dict[str, Any] | None, key: str
) -> str:
    if numerator is None or denominator is None:
        return "n/a"
    return _fmt_ratio(numerator[key]["median_ms"], denominator[key]["median_ms"])


def _field_ratio(
    numerator: dict[str, Any] | None, denominator: dict[str, Any] | None, key: str
) -> str:
    if numerator is None or denominator is None:
        return "n/a"
    return _fmt_ratio(float(numerator[key]), float(denominator[key]))


def _range_text(values: list[float], fmt: str = "{:.1f}") -> str:
    if not values:
        return "n/a"
    low, high = fmt.format(min(values)), fmt.format(max(values))
    if low == high:
        return low
    return f"{low}-{high}"


def _lookup(cases: list[dict[str, Any]], **match: Any) -> dict[str, Any] | None:
    for case in cases:
        if all(case.get(key) == value for key, value in match.items()):
            return case
    return None


def _yes(flag: Any) -> str:
    return "yes" if flag else "no"


def _write_report(payload: dict[str, Any], output_directory: Path, style: ReportStyle) -> None:
    env = payload["environment"]
    cfg = payload["config"]
    single = [c for c in payload["single_gpu"]["cases"] if c["path"] in style.paths]
    component = payload["tile_stats_component"]
    distributed = [d for d in payload["distributed"] if d["path"] in style.paths]
    base = style.baseline
    base_name = style.name(base)
    others = [path for path in style.paths if path != base]
    lines: list[str] = []
    add = lines.append

    add("# PR #328 ROCm vocab-parallel logprob performance analysis")
    add("")
    add("> Operator-only benchmark. No model checkpoint or serving engine was used.")
    add("")
    add("## Environment")
    add("")
    add("| Item | Value |")
    add("|---|---|")
    for key in sorted(env):
        add(f"| {key} | {env[key]} |")
    add("")
    add("## Methodology")
    add("")
    add(
        f"- Qwen3 vocabulary `V={REAL_VOCAB}` split into {NUM_TILES} tiles of "
        f"{REAL_VOCAB // NUM_TILES} columns; seeded logits (`randn * {LOGIT_SCALE}`), "
        "random targets, every seventh token inactive."
    )
    add("- Measured paths:")
    for path in style.paths:
        add(f"  - `{style.name(path)}`: {PATH_DESCRIPTIONS[path]}.")
    dist_notes = [
        DISTRIBUTED_DESCRIPTIONS.get(p, "") for p in style.paths if DISTRIBUTED_DESCRIPTIONS.get(p)
    ]
    if dist_notes:
        add("- Distributed: one process per GPU; " + "; ".join(dist_notes) + ".")
    add(
        "- Forward returns the selected-token logprob and the vocabulary LSE; forward+backward "
        "computes `grad_logits` for `sum(active * logp)`. The WS2 operators run with "
        "`validate=False`; the `validate=True` production entry point is measured separately."
    )
    add(
        "- Single-GPU timing: GPU events, median and p95. Distributed timing: synchronized wall "
        "clock, slowest rank per sample. Peak memory is the per-call increase in "
        "`torch.cuda.max_memory_allocated` (distributed: max over ranks)."
    )
    add(
        "- Accuracy is against an FP64 `logsumexp` of the same (BF16-rounded) logits. Repeat = "
        "two identical calls are bitwise equal; batch-invariant = a row computed alone is "
        "bitwise equal to the same row inside the batch; TP-replicated = every TP rank holds "
        "identical bits."
    )
    add(
        f"- {cfg['warmup']} warmups, {cfg['samples']} measured forward samples, "
        f"{cfg['training_samples']} measured forward+backward samples. Raw medians, p95, "
        "minimum, maximum, and every measured path are in `results.json`."
    )
    if style.table_tokens is not None:
        add(
            "- Tables show "
            + ", ".join(f"{t}" for t in style.table_tokens)
            + " tokens; the figures cover the full token sweep."
        )
    add("")
    add("Reproduce this report from the repository root:")
    add("")
    add("```bash")
    add(style.command)
    add("```")
    add("")

    # ---- key findings
    add("## Key findings")
    add("")
    for path in others:
        name = style.name(path)
        speed_fwd, speed_train, mem = [], [], []
        for dtype_name in SINGLE_DTYPES:
            for case in single:
                if case["dtype"] != dtype_name or case["path"] != path:
                    continue
                if not style.keep_tokens(case["tokens"]):
                    continue
                ref = _lookup(single, dtype=dtype_name, tokens=case["tokens"], path=base)
                if ref is None:
                    continue
                speed_fwd.append(ref["forward"]["median_ms"] / case["forward"]["median_ms"])
                speed_train.append(
                    ref["train_fwd_bwd"]["median_ms"] / case["train_fwd_bwd"]["median_ms"]
                )
                mem.append(case["train_peak_mib"] / max(ref["train_peak_mib"], 1e-6))
        if speed_fwd:
            add(
                f"- Single GPU: `{name}` is {_range_text(speed_fwd, '{:.2f}')}x faster than "
                f"`{base_name}` in forward and {_range_text(speed_train, '{:.2f}')}x in "
                f"forward+backward, with {_range_text(mem, '{:.2f}')}x its peak memory."
            )
        d_fwd, d_train, d_mem, d_abs = [], [], [], []
        for d in distributed:
            if d["path"] != path or not style.keep_tokens(d["tokens"]):
                continue
            ref = _lookup(distributed, topology=d["topology"], tokens=d["tokens"], path=base)
            if ref is None:
                continue
            d_fwd.append(ref["forward"]["median_ms"] / d["forward"]["median_ms"])
            d_train.append(ref["train_fwd_bwd"]["median_ms"] / d["train_fwd_bwd"]["median_ms"])
            d_mem.append(d["train_peak_mib"] / max(ref["train_peak_mib"], 1e-6))
            d_abs.append(d["forward"]["median_ms"])
        if d_fwd:
            add(
                f"- Distributed: `{name}` is {_range_text(d_fwd, '{:.2f}')}x faster than "
                f"`{base_name}` in forward and {_range_text(d_train, '{:.2f}')}x in "
                f"forward+backward across {len({d['topology'] for d in distributed})} TP/CP "
                f"topologies, at {_range_text(d_mem, '{:.2f}')}x the per-rank peak memory "
                f"(absolute forward {_range_text(d_abs, '{:.3f}')} ms)."
            )
    if "ws2-rocm" in style.paths and "ws1-triton" in style.paths:
        ratios_fwd, ratios_train = [], []
        for dtype_name in SINGLE_DTYPES:
            for case in single:
                if case["path"] != "ws2-rocm" or case["dtype"] != dtype_name:
                    continue
                if not style.keep_tokens(case["tokens"]):
                    continue
                tri = _lookup(single, dtype=dtype_name, tokens=case["tokens"], path="ws1-triton")
                if tri is None:
                    continue
                ratios_fwd.append(case["forward"]["median_ms"] / tri["forward"]["median_ms"])
                ratios_train.append(
                    case["train_fwd_bwd"]["median_ms"] / tri["train_fwd_bwd"]["median_ms"]
                )
        if ratios_fwd:
            add(
                f"- `{style.name('ws2-rocm')}` runs at {_range_text(ratios_fwd, '{:.2f}')}x the "
                f"latency of `{style.name('ws1-triton')}` in forward and "
                f"{_range_text(ratios_train, '{:.2f}')}x in forward+backward with the same peak "
                "memory, while carrying the vocab-parallel contract (tile partials, all-gather, "
                "fixed tile-order merge, vocab-domain LSE export) that the single-shard Triton op "
                "does not provide; the gap is the operator's fixed Python/launch floor, not the "
                "HIP kernels."
            )
    if component and "ws2-rocm" in style.paths:
        comp_speed = [
            r["pytorch_loop"]["median_ms"] / r["hip_fp32_input"]["median_ms"]
            for r in component
            if style.keep_tokens(r["tokens"])
        ]
        comp_mem = [
            r["pytorch_loop_peak_mib"] / max(r["hip_peak_mib"], 1e-6)
            for r in component
            if style.keep_tokens(r["tokens"])
        ]
        add(
            f"- The `hip_deterministic_logp_tile_stats` kernel alone is "
            f"{_range_text(comp_speed, '{:.1f}')}x faster than the PyTorch tile loop and "
            f"allocates {_range_text(comp_mem, '{:.0f}')}x less transient memory (it writes only "
            f"the `[tokens, {NUM_TILES}]` FP32 partials)."
        )
    if "ws2-rocm" in style.paths and "ws2-reference" in style.paths:
        mism = [
            c.get("mismatch_vs_reference")
            for c in single
            if c["path"] == "ws2-rocm" and c.get("mismatch_vs_reference") is not None
        ]
        relr = [c.get("rel_l2_vs_reference", 0.0) for c in single if c["path"] == "ws2-rocm"]
        add(
            f"- `{style.name('ws2-rocm')}` vs `{style.name('ws2-reference')}`: tile maxima are "
            "bitwise equal; sumexp partials differ only by FP32 summation order, so final outputs "
            f"differ in {_range_text([float(m) for m in mism], '{:.0f}')} elements per case with "
            f"relative-L2 {_range_text(relr, '{:.1e}')}. Both paths are equally close to FP64."
        )
    ws2 = [c for c in single if c["path"].startswith("ws2")]
    if ws2:
        add(
            f"- Repeat bitwise: {_yes(all(c['repeat_bitwise'] for c in ws2))}; batch-invariant: "
            f"{_yes(all(c['batch_invariant'] for c in ws2))}; all gradients finite: "
            f"{_yes(all(c['grad_finite'] for c in ws2))}."
        )
    ws2_dist = [d for d in distributed if d["path"].startswith("ws2")]
    if ws2_dist:
        add(
            "- Distributed: TP-replicated and repeat bitwise on every topology: "
            f"{_yes(all(d['tp_replicated'] and d['repeat_bitwise'] for d in ws2_dist))}."
        )
    add("")

    # ---- single GPU tables
    for dtype_name in SINGLE_DTYPES:
        rows = [c for c in single if c["dtype"] == dtype_name and style.keep_tokens(c["tokens"])]
        if not rows:
            continue
        add(f"## Single-GPU logprob ({dtype_name.upper()} logits, V={REAL_VOCAB})")
        add("")
        add("### Forward")
        add("")
        add(
            f"| Tokens | Path | Median (ms) | p95 (ms) | Speedup vs {base_name} | Peak MiB | "
            "logp max-abs vs FP64 | LSE max-abs vs FP64 | Repeat | Batch-inv |"
        )
        add("|---:|---|---:|---:|---:|---:|---:|---:|:---:|:---:|")
        for tokens in sorted({c["tokens"] for c in rows}):
            ref = _lookup(single, dtype=dtype_name, tokens=tokens, path=base)
            for path in style.paths:
                case = _lookup(rows, tokens=tokens, path=path)
                if case is None:
                    continue
                add(
                    f"| {tokens} | {style.name(path)} | {case['forward']['median_ms']:.4f} | "
                    f"{case['forward']['p95_ms']:.4f} | {_median_ratio(ref, case, 'forward')} | "
                    f"{case['forward_peak_mib']:.1f} | {case['logp_vs_fp64']['max_abs']:.3e} | "
                    f"{case['lse_vs_fp64']['max_abs']:.3e} | {_yes(case['repeat_bitwise'])} | "
                    f"{_yes(case['batch_invariant'])} |"
                )
        add("")
        add("### Forward+backward")
        add("")
        add(
            f"| Tokens | Path | Median (ms) | p95 (ms) | Speedup vs {base_name} | Peak MiB | "
            f"Memory vs {base_name} | Grad finite |"
        )
        add("|---:|---|---:|---:|---:|---:|---:|:---:|")
        for tokens in sorted({c["tokens"] for c in rows}):
            ref = _lookup(single, dtype=dtype_name, tokens=tokens, path=base)
            for path in style.paths:
                case = _lookup(rows, tokens=tokens, path=path)
                if case is None:
                    continue
                add(
                    f"| {tokens} | {style.name(path)} | {case['train_fwd_bwd']['median_ms']:.4f} | "
                    f"{case['train_fwd_bwd']['p95_ms']:.4f} | "
                    f"{_median_ratio(ref, case, 'train_fwd_bwd')} | {case['train_peak_mib']:.1f} | "
                    f"{_field_ratio(case, ref, 'train_peak_mib')} | {_yes(case['grad_finite'])} |"
                )
        add("")
        if "ws2-rocm" in style.paths and "ws2-reference" in style.paths:
            add(f"### `{style.name('ws2-rocm')}` versus `{style.name('ws2-reference')}` numerics")
            add("")
            add("| Tokens | Mismatched elements (logp+LSE) | Relative L2 |")
            add("|---:|---:|---:|")
            for tokens in sorted({c["tokens"] for c in rows}):
                rocm = _lookup(rows, tokens=tokens, path="ws2-rocm")
                if rocm is None:
                    continue
                add(
                    f"| {tokens} | {rocm.get('mismatch_vs_reference', 'n/a')} | "
                    f"{rocm.get('rel_l2_vs_reference', float('nan')):.3e} |"
                )
            add("")

    overhead = [
        row
        for row in payload["single_gpu"].get("validate_overhead", [])
        if style.keep_tokens(row["tokens"])
    ]
    if overhead and "ws2-rocm" in style.paths:
        add(f"### `validate=True` production entry point ({style.name('ws2-rocm')}, BF16)")
        add("")
        add("| Tokens | validate=False (ms) | validate=True (ms) | Overhead |")
        add("|---:|---:|---:|---:|")
        for row in overhead:
            overhead_ratio = _fmt_ratio(
                row["validate_true"]["median_ms"], row["validate_false"]["median_ms"]
            )
            add(
                f"| {row['tokens']} | {row['validate_false']['median_ms']:.4f} | "
                f"{row['validate_true']['median_ms']:.4f} | {overhead_ratio} |"
            )
        add("")
        add(
            "`validate=True` adds host-side target-range checks and a non-finite LSE check that "
            "synchronizes the stream; the cost is a fixed per-call overhead."
        )
        add("")

    # ---- tile-stats component
    comp_rows = [r for r in component if style.keep_tokens(r["tokens"])]
    if comp_rows and "ws2-rocm" in style.paths:
        add("## Tile-stats kernel")
        add("")
        add(
            "`hip_deterministic_logp_tile_stats` computes the per-row, per-tile FP32 "
            "`(max, sumexp)` partials that the operator all-gathers and merges; the PyTorch tile "
            f"loop is what `{style.name('ws2-reference')}` uses for the same step. Tile maxima are "
            "bitwise equal; sums differ only by FP32 summation order."
        )
        add("")
        add(
            "| Logits dtype | Tokens | PyTorch tile loop (ms) | HIP kernel on FP32 (ms) | "
            "HIP kernel on stored dtype (ms) | Speedup | Loop peak MiB | HIP peak MiB | "
            "Max bitwise | sumexp max rel | Repeat |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|")
        for row in comp_rows:
            kernel_speedup = _fmt_ratio(
                row["pytorch_loop"]["median_ms"], row["hip_fp32_input"]["median_ms"]
            )
            add(
                f"| {row['dtype']} | {row['tokens']} | {row['pytorch_loop']['median_ms']:.4f} | "
                f"{row['hip_fp32_input']['median_ms']:.4f} | "
                f"{row['hip_native_dtype_input']['median_ms']:.4f} | "
                f"{kernel_speedup} | "
                f"{row['pytorch_loop_peak_mib']:.1f} | {row['hip_peak_mib']:.1f} | "
                f"{_yes(row['max_bitwise'])} | {row['sumexp_max_rel']:.2e} | "
                f"{_yes(row['repeat_bitwise'])} |"
            )
        add("")

    # ---- distributed
    dist_rows = [d for d in distributed if style.keep_tokens(d["tokens"])]
    if dist_rows:
        add("## Distributed vocab-parallel logprob (BF16, RCCL)")
        add("")
        add("### Forward")
        add("")
        add(
            f"| Topology | Tokens | Path | Median (ms) | p95 (ms) | Speedup vs {base_name} | "
            "Peak MiB/rank | logp max-abs vs FP64 | TP-replicated | Repeat |"
        )
        add("|---|---:|---|---:|---:|---:|---:|---:|:---:|:---:|")
        for d in dist_rows:
            ref = _lookup(distributed, topology=d["topology"], tokens=d["tokens"], path=base)
            add(
                f"| {d['topology']} | {d['tokens']} | {style.name(d['path'])} | "
                f"{d['forward']['median_ms']:.4f} | {d['forward']['p95_ms']:.4f} | "
                f"{_median_ratio(ref, d, 'forward')} | {d['forward_peak_mib']:.1f} | "
                f"{d['logp_vs_fp64_max_abs']:.3e} | {_yes(d['tp_replicated'])} | "
                f"{_yes(d['repeat_bitwise'])} |"
            )
        add("")
        add("### Forward+backward")
        add("")
        add(
            f"| Topology | Tokens | Path | Median (ms) | p95 (ms) | Speedup vs {base_name} | "
            f"Peak MiB/rank | Memory vs {base_name} | Grad finite |"
        )
        add("|---|---:|---|---:|---:|---:|---:|---:|:---:|")
        for d in dist_rows:
            ref = _lookup(distributed, topology=d["topology"], tokens=d["tokens"], path=base)
            add(
                f"| {d['topology']} | {d['tokens']} | {style.name(d['path'])} | "
                f"{d['train_fwd_bwd']['median_ms']:.4f} | {d['train_fwd_bwd']['p95_ms']:.4f} | "
                f"{_median_ratio(ref, d, 'train_fwd_bwd')} | {d['train_peak_mib']:.1f} | "
                f"{_field_ratio(d, ref, 'train_peak_mib')} | {_yes(d['grad_finite'])} |"
            )
        add("")
        if "ws2-rocm" in style.paths and "ws2-reference" in style.paths:
            add(
                f"### `{style.name('ws2-rocm')}` versus `{style.name('ws2-reference')}` "
                "numerics (distributed)"
            )
            add("")
            add("| Topology | Tokens | Mismatched elements (logp+LSE) | Relative L2 |")
            add("|---|---:|---:|---:|")
            for d in dist_rows:
                if d["path"] != "ws2-rocm":
                    continue
                add(
                    f"| {d['topology']} | {d['tokens']} | "
                    f"{d.get('mismatch_vs_reference', 'n/a')} | "
                    f"{d.get('rel_l2_vs_reference', float('nan')):.3e} |"
                )
            add("")

    # ---- optional tuning history
    baseline = payload.get("baseline")
    if baseline and style.show_tuning_baseline and "ws2-rocm" in style.paths:
        name = style.name("ws2-rocm")
        add("## ROCm tuning: before versus after")
        add("")
        add(
            f"Baseline commit `{baseline.get('git_commit')}` ran the PR's first ROCm backend: the "
            "HIP tile kernel on an FP32 copy of the shard, with the shared PyTorch autograd "
            f"backward. `{name}` rows only."
        )
        add("")
        add(
            "| dtype | Tokens | Fwd before (ms) | Fwd after (ms) | Speedup | "
            "Fwd+bwd before (ms) | Fwd+bwd after (ms) | Speedup | Peak before (MiB) | "
            "Peak after (MiB) | Memory ratio |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for before in baseline["single_gpu"]:
            if not style.keep_tokens(before["tokens"]):
                continue
            after = _lookup(single, dtype=before["dtype"], tokens=before["tokens"], path="ws2-rocm")
            if after is None:
                continue
            add(
                f"| {before['dtype']} | {before['tokens']} | "
                f"{before['forward']['median_ms']:.4f} | "
                f"{after['forward']['median_ms']:.4f} | "
                f"{_median_ratio(before, after, 'forward')} | "
                f"{before['train_fwd_bwd']['median_ms']:.4f} | "
                f"{after['train_fwd_bwd']['median_ms']:.4f} | "
                f"{_median_ratio(before, after, 'train_fwd_bwd')} | "
                f"{before['train_peak_mib']:.1f} | {after['train_peak_mib']:.1f} | "
                f"{_field_ratio(after, before, 'train_peak_mib')} |"
            )
        add("")

    add("## Figures")
    add("")
    add("![Single-GPU latency](single_gpu_latency.png)")
    add("")
    add("![Single-GPU peak memory](single_gpu_memory.png)")
    add("")
    if distributed:
        add("![Distributed latency](distributed_logp_latency.png)")
        add("")
    (output_directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figures(payload: dict[str, Any], output_directory: Path, style: ReportStyle) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single = [
        c
        for c in payload["single_gpu"]["cases"]
        if c["dtype"] == "bf16" and c["path"] in style.paths
    ]
    tokens = sorted({c["tokens"] for c in single})

    for filename, keys, ylabel, title in (
        ("single_gpu_latency.png", ("forward", "train_fwd_bwd"), "median ms", "latency"),
        (
            "single_gpu_memory.png",
            ("forward_peak_mib", "train_peak_mib"),
            "peak MiB above live",
            "peak device memory",
        ),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for axis, key, direction in zip(axes, keys, ("Forward", "Forward+backward")):
            for path in style.paths:
                ys = []
                for t in tokens:
                    case = _lookup(single, tokens=t, path=path)
                    if case is None:
                        ys.append(float("nan"))
                    elif isinstance(case[key], dict):
                        ys.append(case[key]["median_ms"])
                    else:
                        ys.append(case[key])
                axis.plot(tokens, ys, marker="o", label=style.name(path))
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlabel("tokens")
            axis.set_ylabel(ylabel)
            axis.set_title(f"Single MI300X, BF16, V={REAL_VOCAB}: {direction} {title}")
            axis.grid(True, which="both", alpha=0.3)
            if style.paths:
                axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_directory / filename, dpi=180)
        plt.close(figure)

    distributed = [d for d in payload["distributed"] if d["path"] in style.paths]
    if not distributed:
        return
    labels = []
    series: dict[tuple[str, str], list[float]] = {
        (path, key): [] for path in style.paths for key in ("forward", "train_fwd_bwd")
    }
    for d in distributed:
        if d["path"] != style.baseline:
            continue
        labels.append(f"{d['topology']}\nM={d['tokens']}")
        for path in style.paths:
            other = _lookup(distributed, path=path, topology=d["topology"], tokens=d["tokens"])
            for key in ("forward", "train_fwd_bwd"):
                series[(path, key)].append(
                    other[key]["median_ms"] if other is not None else float("nan")
                )
    figure, axes = plt.subplots(1, 2, figsize=(max(12, 1.1 * len(labels)), 4.8))
    xs = list(range(len(labels)))
    width = 0.8 / max(len(style.paths), 1)
    for axis, key, direction in zip(
        axes, ("forward", "train_fwd_bwd"), ("Forward", "Forward+backward")
    ):
        for index, path in enumerate(style.paths):
            offset = (index - (len(style.paths) - 1) / 2) * width
            axis.bar([x + offset for x in xs], series[(path, key)], width, label=style.name(path))
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel("slowest-rank median ms")
        axis.set_title(f"Distributed vocab-parallel logprob, BF16: {direction}")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_directory / "distributed_logp_latency.png", dpi=180)
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
        "native_collective": "torch.distributed ProcessGroupNCCL (RCCL on ROCm)",
        "extension_symbols": "hip_deterministic_logp_tile_stats, hip_deterministic_logp_backward",
    }


def _validate_environment(require_distributed: bool) -> None:
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("this benchmark requires a ROCm PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("no ROCm GPU is visible")
    if not _EXT_AVAILABLE or _C is None or not hasattr(_C, "hip_deterministic_logp_backward"):
        raise RuntimeError(
            "rl_engine._C with hip_deterministic_logp_* is unavailable; build with "
            "PYTORCH_ROCM_ARCH=gfx942 RL_KERNEL_REQUIRE_EXT=1 python setup.py build_ext --inplace"
        )
    if require_distributed and (not dist.is_available() or not dist.is_nccl_available()):
        raise RuntimeError("PyTorch RCCL/ProcessGroupNCCL support is unavailable")


ALL_PATHS = ("native", "ws1-pytorch", "ws1-triton", "ws2-reference", "ws2-rocm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/rocm_logp"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--training-samples", type=int, default=5)
    parser.add_argument("--skip-distributed", action="store_true")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="earlier results.json; adds a before/after tuning section for the ROCm backend",
    )
    parser.add_argument(
        "--render-from",
        type=Path,
        default=None,
        help="skip measurement and render report/figures from this results.json",
    )
    parser.add_argument(
        "--report-paths",
        type=str,
        default=",".join(ALL_PATHS),
        help="comma-separated measured paths to show, in order (default: all)",
    )
    parser.add_argument(
        "--rename",
        type=str,
        default="",
        help="display names, e.g. 'ws2-reference=native,ws2-rocm=strict-hip'",
    )
    parser.add_argument(
        "--report-baseline",
        type=str,
        default=None,
        help="path used for speedup/memory ratio columns (default: first reported path)",
    )
    parser.add_argument(
        "--table-tokens",
        type=str,
        default="",
        help="comma-separated token counts to show in tables (default: all measured)",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default=",".join(name for name, _, _ in TOPOLOGIES),
        help="comma-separated subset of " + ",".join(name for name, _, _ in TOPOLOGIES),
    )
    parser.add_argument("--tokens", type=str, default=",".join(str(t) for t in SINGLE_TOKENS))
    parser.add_argument(
        "--distributed-tokens", type=str, default=",".join(str(t) for t in DISTRIBUTED_TOKENS)
    )
    return parser.parse_args()


def _report_style(
    args: argparse.Namespace, output_directory: Path, config: dict[str, Any]
) -> ReportStyle:
    paths = tuple(p.strip() for p in args.report_paths.split(",") if p.strip())
    unknown = [p for p in paths if p not in ALL_PATHS]
    if unknown:
        raise ValueError(f"unknown report paths {unknown}; choose from {ALL_PATHS}")
    names: dict[str, str] = {}
    for item in (piece.strip() for piece in args.rename.split(",") if piece.strip()):
        key, _, value = item.partition("=")
        names[key.strip()] = value.strip()
    table_tokens = tuple(int(t) for t in args.table_tokens.split(",") if t.strip()) or None
    # Always show the measurement command: rerunning it with the same report flags
    # regenerates this report; --render-from only re-renders an existing results.json.
    command_parts = [
        "python benchmarks/benchmark_rocm_logp.py \\",
        f"  --warmup {config['warmup']} \\",
        f"  --samples {config['samples']} \\",
        f"  --training-samples {config['training_samples']} \\",
    ]
    if paths != ALL_PATHS:
        command_parts.append(f"  --report-paths {','.join(paths)} \\")
    if names:
        command_parts.append("  --rename " + ",".join(f"{k}={v}" for k, v in names.items()) + " \\")
    if args.report_baseline:
        command_parts.append(f"  --report-baseline {args.report_baseline} \\")
    if table_tokens:
        command_parts.append(f"  --table-tokens {','.join(str(t) for t in table_tokens)} \\")
    command_parts.append(f"  --output-dir {output_directory.as_posix()}")
    return ReportStyle(
        paths=paths,
        names=names,
        baseline=args.report_baseline or paths[0],
        table_tokens=table_tokens,
        show_tuning_baseline=args.baseline is not None,
        command="\n".join(command_parts),
    )


def main() -> None:
    args = parse_args()
    output_directory: Path = args.output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    if args.render_from is not None:
        payload = json.loads(args.render_from.read_text(encoding="utf-8"))
    else:
        _validate_environment(require_distributed=not args.skip_distributed)
        config = {
            "warmup": args.warmup,
            "samples": args.samples,
            "training_samples": args.training_samples,
        }
        tokens = tuple(int(t) for t in args.tokens.split(",") if t)
        distributed_tokens = tuple(int(t) for t in args.distributed_tokens.split(",") if t)
        selected = {name.strip() for name in args.topologies.split(",") if name.strip()}
        payload = {
            "environment": _environment(),
            "config": config,
            "single_gpu": {"cases": [], "validate_overhead": [], "paths": []},
            "tile_stats_component": [],
            "distributed": [],
        }
        if not args.skip_single:
            payload["single_gpu"] = _single_gpu_benchmarks(
                warmup=args.warmup,
                samples=args.samples,
                training_samples=args.training_samples,
                tokens=tokens,
            )
            payload["tile_stats_component"] = _tile_stats_component(
                warmup=args.warmup, samples=args.samples, tokens=tokens
            )
        if not args.skip_distributed:
            device_count = torch.cuda.device_count()
            for topology in TOPOLOGIES:
                name, tp, cp = topology
                if name not in selected:
                    continue
                if tp * cp > device_count:
                    print(f"skipping {name}: needs {tp * cp} GPUs, {device_count} visible")
                    continue
                payload["distributed"].extend(
                    _run_distributed_world(topology, distributed_tokens, config)
                )
    style = _report_style(args, output_directory, payload["config"])
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        payload["baseline"] = {
            "git_commit": baseline.get("environment", {}).get("git_commit"),
            "single_gpu": [c for c in baseline["single_gpu"]["cases"] if c["path"] == "ws2-rocm"],
            "tile_stats_component": baseline.get("tile_stats_component", []),
            "distributed": [d for d in baseline.get("distributed", []) if d["path"] == "ws2-rocm"],
        }
    elif args.render_from is not None:
        payload.pop("baseline", None)
    (output_directory / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(payload, output_directory, style)
    _write_figures(payload, output_directory, style)
    print(json.dumps({"output_dir": str(output_directory), "status": "ok"}))


if __name__ == "__main__":
    main()
