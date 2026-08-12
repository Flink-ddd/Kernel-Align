# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bias-free gated FFN backward assembled from deterministic CUDA kernels."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

QWEN3_8B_HIDDEN_SIZE = 4096
QWEN3_8B_INTERMEDIATE_SIZE = 12288

_REQUIRED_SYMBOLS = (
    "det_gemm_fwd",
    "det_gemm_db",
    "swiglu_backward",
)


def _require_ffn_backward_kernels() -> None:
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(_C, name)]
    if not _EXT_AVAILABLE or _C is None or missing:
        suffix = f" Missing symbols: {', '.join(missing)}." if missing else ""
        raise RuntimeError(
            "qwen3_ffn_backward requires the compiled deterministic GEMM and "
            f"SwiGLU CUDA kernels.{suffix}"
        )


def _require_tensor_parallel_group(tp_group: Any):
    if tp_group is None:
        return None

    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError("tensor-parallel FFN backward requires torch.distributed.")
    if not dist.is_initialized():
        raise RuntimeError("tensor-parallel FFN backward requires an initialized process group.")
    if dist.get_world_size(group=tp_group) <= 1:
        raise ValueError("tp_group must contain at least two ranks.")
    return dist


def _validate_ffn_backward_inputs(
    grad_output: Tensor,
    rmsnorm_output: Tensor,
    gate: Tensor,
    up: Tensor,
    activated: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> None:
    tensors = {
        "grad_output": grad_output,
        "rmsnorm_output": rmsnorm_output,
        "gate": gate,
        "up": up,
        "activated": activated,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "down_weight": down_weight,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}.")

    if rmsnorm_output.dim() < 1:
        raise ValueError("rmsnorm_output must have at least one dimension.")
    if rmsnorm_output.numel() == 0:
        raise ValueError("rmsnorm_output must contain at least one token.")
    for name, weight in (
        ("gate_weight", gate_weight),
        ("up_weight", up_weight),
        ("down_weight", down_weight),
    ):
        if weight.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got shape {tuple(weight.shape)}.")

    hidden_size = rmsnorm_output.size(-1)
    intermediate_size = gate.size(-1)
    expected_shapes = {
        "grad_output": (*rmsnorm_output.shape[:-1], hidden_size),
        "gate": (*rmsnorm_output.shape[:-1], intermediate_size),
        "up": (*rmsnorm_output.shape[:-1], intermediate_size),
        "activated": (*rmsnorm_output.shape[:-1], intermediate_size),
        "gate_weight": (intermediate_size, hidden_size),
        "up_weight": (intermediate_size, hidden_size),
        "down_weight": (hidden_size, intermediate_size),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}.")

    for name, tensor in tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype bfloat16, got {tensor.dtype}.")
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be on a CUDA device, got '{tensor.device}'.")
        if tensor.device != rmsnorm_output.device:
            raise RuntimeError(
                f"all FFN inputs must be on {rmsnorm_output.device}, "
                f"got {name} on {tensor.device}."
            )


def qwen3_ffn_backward(
    grad_output: Tensor,
    rmsnorm_output: Tensor,
    gate: Tensor,
    up: Tensor,
    activated: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    tp_group: Any = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute all gradients of a bias-free SiLU-gated FFN.

    ``T`` denotes the product of all leading token dimensions, ``H`` the hidden
    width, and ``I`` the intermediate width. The corresponding forward is::

        gate = rmsnorm_output @ gate_weight.T
        up = rmsnorm_output @ up_weight.T
        activated = silu(gate) * up
        output = activated @ down_weight.T

    Args:
        grad_output: Gradient from the FFN output, shape ``[..., H]``.
        rmsnorm_output: Saved RMSNorm output consumed by both input projections,
            shape ``[..., H]``. It is the tensor whose gradient must be returned
            to the RMSNorm backward.
        gate: Saved gate projection output before SiLU, shape ``[..., I]``.
        up: Saved up projection output, shape ``[..., I]``.
        activated: Saved SwiGLU result ``silu(gate) * up``, shape ``[..., I]``.
        gate_weight: Gate projection weight in ``[out, in]`` layout, shape
            ``[I, H]``.
        up_weight: Up projection weight in ``[out, in]`` layout, shape
            ``[I, H]``.
        down_weight: Down projection weight in ``[out, in]`` layout, shape
            ``[H, I]``.
        tp_group: Optional tensor-parallel process group. ``None`` selects the
            single-rank path.

    TP gradient layout (``I_local = I / TP``):
        - Gate/Up are column-parallel. Their input gradients have the same
          ``[T,H]`` coordinates and are reduced across TP. Their weight
          gradients are shards that concatenate on dimension 0 to ``[I,H]``.
        - Down is row-parallel. Its input-gradient shards concatenate on the
          last dimension to ``[T,I]``. Its weight-gradient shards concatenate
          on dimension 1 to ``[H,I]``.

        This function performs the Gate and Up input-gradient reductions
        separately, then adds the two complete ``[T,H]`` gradients locally.
        Concatenation describes shard ownership and does not require a
        collective inside this function.

    Returns:
        ``(grad_rmsnorm_output, grad_gate_weight, grad_up_weight,
        grad_down_weight)``. Each gradient has the same shape and layout as its
        corresponding forward input.
    """

    _validate_ffn_backward_inputs(
        grad_output,
        rmsnorm_output,
        gate,
        up,
        activated,
        gate_weight,
        up_weight,
        down_weight,
    )
    _require_ffn_backward_kernels()
    dist = _require_tensor_parallel_group(tp_group)

    # GEMM kernels accept 2-D matrices. Flatten all leading token dimensions
    # into T while preserving H or I as the reduction/projection dimension.
    rmsnorm_output_2d = rmsnorm_output.reshape(-1, rmsnorm_output.size(-1)).contiguous()
    grad_output_2d = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
    gate_2d = gate.reshape(-1, gate.size(-1)).contiguous()
    up_2d = up.reshape(-1, up.size(-1)).contiguous()
    activated_2d = activated.reshape(-1, activated.size(-1)).contiguous()

    # Down is row-parallel. Each rank returns grad_down_weight[H,I_local]; the
    # full weight gradient is concat(local_grad, dim=1).
    grad_down_weight = _C.det_gemm_db(activated_2d, grad_output_2d).t().contiguous()

    # Each rank returns grad_activated[T,I_local]; the full input gradient of
    # Down is concat(local_grad, dim=-1), so no TP reduction is needed here.
    grad_activated = _C.det_gemm_fwd(grad_output_2d, down_weight)

    # Differentiate activated = silu(gate) * up elementwise.
    grad_gate, grad_up = _C.swiglu_backward(grad_activated, gate_2d, up_2d)

    # Gate/Up are column-parallel. Each local weight gradient is [I_local,H];
    # the full weight gradient is concat(local_grad, dim=0).
    grad_gate_weight = _C.det_gemm_db(rmsnorm_output_2d, grad_gate).t().contiguous()
    grad_up_weight = _C.det_gemm_db(rmsnorm_output_2d, grad_up).t().contiguous()

    # Gate/Up local input gradients address the same [T,H] coordinates. Keep
    # their ColumnParallel backward boundaries separate: reduce each complete
    # branch across TP, then add the two results locally.
    grad_rmsnorm_from_gate = _C.det_gemm_fwd(grad_gate, gate_weight)
    if dist is not None:
        # TODO: CUDA currently uses NCCL. Replace it with the custom
        # deterministic AllReduce and compare both communication paths.
        dist.all_reduce(grad_rmsnorm_from_gate, op=dist.ReduceOp.SUM, group=tp_group)

    grad_rmsnorm_from_up = _C.det_gemm_fwd(grad_up, up_weight)
    if dist is not None:
        dist.all_reduce(grad_rmsnorm_from_up, op=dist.ReduceOp.SUM, group=tp_group)

    grad_rmsnorm_output = grad_rmsnorm_from_gate.add_(grad_rmsnorm_from_up)

    return (
        grad_rmsnorm_output.reshape_as(rmsnorm_output),
        grad_gate_weight,
        grad_up_weight,
        grad_down_weight,
    )
