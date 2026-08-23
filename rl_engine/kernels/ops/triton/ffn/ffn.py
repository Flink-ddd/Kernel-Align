# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Distributed deterministic Qwen3 FFN assembled from ROCm Triton kernels.

The arithmetic and sharding contract matches the native PR #325 path exactly:
BF16 is preserved at every GEMM/SwiGLU boundary, GEMMs use the canonical
FP32-leaf/BF16-node K tree, and TP reductions use the rank-ordered balanced
collective.  CP gathers full token sequences before weight-gradient GEMMs so
their K tree is identical to CP=1.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.ops.pytorch.ffn.ffn import (
    _all_gather_tokens,
    _all_reduce_inplace,
    _collective_for_group,
    _reduce_scatter_tokens,
    _require_parallel_group,
    _validate_ffn_inputs,
)
from rl_engine.kernels.ops.triton.activation.swiglu import (
    _launch_swiglu_bwd,
    _launch_swiglu_fwd,
)
from rl_engine.kernels.ops.triton.matmul.det_gemm import _triton_tree_gemm


def _gemm(a: Tensor, b: Tensor) -> Tensor:
    return _triton_tree_gemm(a, b)


def _gemm_db(a: Tensor, grad_output: Tensor) -> Tensor:
    return _triton_tree_gemm(a.t().contiguous(), grad_output)


class _TritonDeterministicFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        rmsnorm_output: Tensor,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        tp_group: Any,
        cp_group: Any,
        sequence_parallel: bool,
    ) -> Tensor:
        tp_dist = _require_parallel_group(tp_group, "tensor")
        _require_parallel_group(cp_group, "context")
        if sequence_parallel and tp_dist is None:
            raise ValueError("sequence_parallel requires a tensor-parallel group.")

        input_shape = rmsnorm_output.shape
        rmsnorm_output_2d = rmsnorm_output.reshape(-1, input_shape[-1]).contiguous()
        tp_world = tp_dist.get_world_size(group=tp_group) if tp_dist is not None else 1
        gemm_tokens = rmsnorm_output_2d.size(0) * (tp_world if sequence_parallel else 1)
        element_size = rmsnorm_output_2d.element_size()
        min_size_bytes = max(
            gemm_tokens * rmsnorm_output_2d.size(1) * element_size,
            gemm_tokens * gate_weight.size(0) * element_size,
            gate_weight.numel() * element_size,
            up_weight.numel() * element_size,
            down_weight.numel() * element_size,
        )
        tp_collective = _collective_for_group(tp_group, min_size_bytes=min_size_bytes)
        cp_collective = _collective_for_group(cp_group, min_size_bytes=min_size_bytes)

        if sequence_parallel:
            rmsnorm_output_2d = _all_gather_tokens(rmsnorm_output_2d, tp_collective)

        gate = _gemm(rmsnorm_output_2d, gate_weight.t().contiguous())
        up = _gemm(rmsnorm_output_2d, up_weight.t().contiguous())
        activated = _launch_swiglu_fwd(gate, up)
        output = _gemm(activated, down_weight.t().contiguous())

        if sequence_parallel:
            output = _reduce_scatter_tokens(output, tp_collective)
        elif tp_collective is not None:
            output = _all_reduce_inplace(output, tp_collective)

        ctx.save_for_backward(
            rmsnorm_output_2d,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        )
        ctx.input_shape = input_shape
        ctx.tp_collective = tp_collective
        ctx.cp_collective = cp_collective
        ctx.sequence_parallel = sequence_parallel
        return output.reshape(*input_shape[:-1], output.size(-1))

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:
        (
            rmsnorm_output,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        ) = ctx.saved_tensors
        tp_collective = ctx.tp_collective
        cp_collective = ctx.cp_collective
        grad_output = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
        if ctx.sequence_parallel:
            grad_output = _all_gather_tokens(grad_output, tp_collective)

        if cp_collective is not None:
            activated_full = _all_gather_tokens(activated, cp_collective)
            grad_output_full = _all_gather_tokens(grad_output, cp_collective)
            grad_down_weight = _gemm_db(activated_full, grad_output_full).t().contiguous()
        else:
            grad_down_weight = _gemm_db(activated, grad_output).t().contiguous()

        grad_activated = _gemm(grad_output, down_weight)
        grad_gate, grad_up = _launch_swiglu_bwd(grad_activated, gate, up)

        if cp_collective is not None:
            rmsnorm_full = _all_gather_tokens(rmsnorm_output, cp_collective)
            grad_gate_full = _all_gather_tokens(grad_gate, cp_collective)
            grad_up_full = _all_gather_tokens(grad_up, cp_collective)
            grad_gate_weight = _gemm_db(rmsnorm_full, grad_gate_full).t().contiguous()
            grad_up_weight = _gemm_db(rmsnorm_full, grad_up_full).t().contiguous()
        else:
            grad_gate_weight = _gemm_db(rmsnorm_output, grad_gate).t().contiguous()
            grad_up_weight = _gemm_db(rmsnorm_output, grad_up).t().contiguous()

        grad_rmsnorm_from_gate = _gemm(grad_gate, gate_weight)
        if ctx.sequence_parallel:
            grad_rmsnorm_from_gate = _reduce_scatter_tokens(
                grad_rmsnorm_from_gate,
                tp_collective,
            )
        elif tp_collective is not None:
            grad_rmsnorm_from_gate = _all_reduce_inplace(
                grad_rmsnorm_from_gate,
                tp_collective,
            )

        grad_rmsnorm_from_up = _gemm(grad_up, up_weight)
        if ctx.sequence_parallel:
            grad_rmsnorm_from_up = _reduce_scatter_tokens(
                grad_rmsnorm_from_up,
                tp_collective,
            )
        elif tp_collective is not None:
            grad_rmsnorm_from_up = _all_reduce_inplace(
                grad_rmsnorm_from_up,
                tp_collective,
            )

        grad_rmsnorm_output = grad_rmsnorm_from_gate.add_(grad_rmsnorm_from_up)
        return (
            grad_rmsnorm_output.reshape(ctx.input_shape),
            grad_gate_weight,
            grad_up_weight,
            grad_down_weight,
            None,
            None,
            None,
        )


def qwen3_ffn_triton(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    tp_group: Any = None,
    cp_group: Any = None,
    sequence_parallel: bool = False,
) -> Tensor:
    """Apply the distributed deterministic Qwen3 FFN with Triton kernels."""

    _validate_ffn_inputs(rmsnorm_output, gate_weight, up_weight, down_weight)
    if not isinstance(sequence_parallel, bool):
        raise TypeError(
            f"sequence_parallel must be a bool, got {type(sequence_parallel)!r}."
        )
    return _TritonDeterministicFFNFunction.apply(
        rmsnorm_output,
        gate_weight,
        up_weight,
        down_weight,
        tp_group,
        cp_group,
        sequence_parallel,
    )
