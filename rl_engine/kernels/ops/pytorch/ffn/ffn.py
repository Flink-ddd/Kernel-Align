# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bias-free gated FFN assembled from deterministic GPU kernels.

The numerical contract intentionally follows PR #321: GEMM and SwiGLU each
consume and produce BF16 tensors.  This module does not keep the whole FFN in
FP32 and downcast only its final result.
"""

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
    "swiglu_forward",
    "swiglu_backward",
)
_COLLECTIVE_MIN_CAPACITY_BYTES = 64 * 1024 * 1024
_COLLECTIVES: dict[tuple[int, int, int, int], Any] = {}


def _require_ffn_kernels() -> None:
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(_C, name)]
    if not _EXT_AVAILABLE or _C is None or missing:
        suffix = f" Missing symbols: {', '.join(missing)}." if missing else ""
        raise RuntimeError(
            f"qwen3_ffn requires the compiled deterministic GEMM and SwiGLU GPU kernels.{suffix}"
        )


def _require_parallel_group(group: Any, name: str):
    if group is None:
        return None

    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError(f"{name}-parallel FFN requires torch.distributed.")
    if not dist.is_initialized():
        raise RuntimeError(f"{name}-parallel FFN requires an initialized process group.")
    if dist.get_world_size(group=group) <= 1:
        raise ValueError(f"{name}_group must contain at least two ranks.")
    return dist


def _validate_ffn_inputs(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> None:
    tensors = {
        "rmsnorm_output": rmsnorm_output,
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
    intermediate_size = gate_weight.size(0)
    if intermediate_size == 0:
        raise ValueError("FFN intermediate size must be positive.")
    expected_shapes = {
        "gate_weight": (intermediate_size, hidden_size),
        "up_weight": (intermediate_size, hidden_size),
        "down_weight": (hidden_size, intermediate_size),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}.")

    # Validate dtype for every input before device placement so callers get
    # the most actionable contract error even on CPU-only build hosts.
    for name, tensor in tensors.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype bfloat16, got {tensor.dtype}.")

    for name, tensor in tensors.items():
        # PyTorch exposes both NVIDIA CUDA and AMD ROCm devices through the
        # ``cuda`` device API, so ``is_cuda`` is correct for both backends.
        if not tensor.is_cuda:
            raise RuntimeError(f"{name} must be on a CUDA/ROCm GPU device, got '{tensor.device}'.")
        if tensor.device != rmsnorm_output.device:
            raise RuntimeError(
                f"all FFN inputs must be on {rmsnorm_output.device}, got {name} on {tensor.device}."
            )


def _create_collective(*, group: Any, max_size_bytes: int):
    """Create the platform collective lazily, keeping single-GPU independent.

    This small boundary is intentionally easy to inject in unit tests.  The
    distributed factory selects CUDA IPC on NVIDIA and RCCL transport with a
    deterministic local reduction tree on ROCm.
    """

    try:
        from rl_engine.distributed import create_deterministic_collective
    except ImportError as exc:
        raise RuntimeError(
            "parallel qwen3_ffn requires the platform deterministic collective "
            "factory; single-device qwen3_ffn remains available"
        ) from exc

    return create_deterministic_collective(
        group=group,
        max_size_bytes=max_size_bytes,
    )


def _collective_for_group(group: Any, *, min_size_bytes: int):
    if group is None:
        return None

    import torch.distributed as dist

    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    device_index = torch.cuda.current_device()
    key = (id(group), rank, world_size, device_index)
    cached = _COLLECTIVES.get(key)
    if cached is not None and cached.max_size_bytes >= min_size_bytes:
        return cached
    if cached is not None:
        cached.close()

    collective = _create_collective(
        group=group,
        max_size_bytes=max(_COLLECTIVE_MIN_CAPACITY_BYTES, min_size_bytes),
    )
    _COLLECTIVES[key] = collective
    return collective


def _all_gather_tokens(tensor: Tensor, collective: Any) -> Tensor:
    return collective.all_gather(tensor.contiguous())


def _reduce_scatter_tokens(tensor: Tensor, collective: Any) -> Tensor:
    world_size = collective.world_size
    if tensor.size(0) % world_size != 0:
        raise ValueError(
            "the gathered token count must be divisible by the tensor-parallel "
            f"world size, got {tensor.size(0)} and {world_size}."
        )
    return collective.reduce_scatter(tensor.contiguous())


def _all_reduce_inplace(tensor: Tensor, collective: Any) -> Tensor:
    return collective.all_reduce(tensor, out=tensor)


class _DeterministicFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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
        # Create TP before CP so every rank follows the same group order.
        tp_collective = _collective_for_group(tp_group, min_size_bytes=min_size_bytes)
        cp_collective = _collective_for_group(cp_group, min_size_bytes=min_size_bytes)

        if sequence_parallel:
            rmsnorm_output_2d = _all_gather_tokens(rmsnorm_output_2d, tp_collective)

        # Model weights use HuggingFace's [out, in] layout; GEMM consumes [K, N].
        # Each native call returns BF16 before the next stage, matching PR #321.
        gate = _C.det_gemm_fwd(rmsnorm_output_2d, gate_weight.t().contiguous())
        up = _C.det_gemm_fwd(rmsnorm_output_2d, up_weight.t().contiguous())
        activated = _C.swiglu_forward(gate, up)
        output = _C.det_gemm_fwd(activated, down_weight.t().contiguous())

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
    def backward(ctx, grad_output: Tensor):
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

        # Down weight gradients must see every CP token so det_gemm_db's
        # reduction tree matches CP=1. Local dW plus AllReduce would use a
        # different parenthesization at many token counts.
        if cp_collective is not None:
            activated_full = _all_gather_tokens(activated, cp_collective)
            grad_output_full = _all_gather_tokens(grad_output, cp_collective)
            grad_down_weight = _C.det_gemm_db(activated_full, grad_output_full).t().contiguous()
        else:
            grad_down_weight = _C.det_gemm_db(activated, grad_output).t().contiguous()

        # Down input-gradient shards concatenate across TP; no TP reduction.
        grad_activated = _C.det_gemm_fwd(grad_output, down_weight)
        grad_gate, grad_up = _C.swiglu_backward(grad_activated, gate, up)

        if cp_collective is not None:
            rmsnorm_full = _all_gather_tokens(rmsnorm_output, cp_collective)
            grad_gate_full = _all_gather_tokens(grad_gate, cp_collective)
            grad_up_full = _all_gather_tokens(grad_up, cp_collective)
            grad_gate_weight = _C.det_gemm_db(rmsnorm_full, grad_gate_full).t().contiguous()
            grad_up_weight = _C.det_gemm_db(rmsnorm_full, grad_up_full).t().contiguous()
        else:
            grad_gate_weight = _C.det_gemm_db(rmsnorm_output, grad_gate).t().contiguous()
            grad_up_weight = _C.det_gemm_db(rmsnorm_output, grad_up).t().contiguous()

        # Gate/Up input gradients reduce across TP, then add locally.
        grad_rmsnorm_from_gate = _C.det_gemm_fwd(grad_gate, gate_weight)
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

        grad_rmsnorm_from_up = _C.det_gemm_fwd(grad_up, up_weight)
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


def qwen3_ffn(
    rmsnorm_output: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    tp_group: Any = None,
    cp_group: Any = None,
    sequence_parallel: bool = False,
) -> Tensor:
    """Apply a bias-free SiLU-gated FFN with deterministic backward kernels.

    Args:
        rmsnorm_output: RMSNorm output, shape ``[..., H]``.
        gate_weight: Gate projection in HuggingFace ``[out, in]`` layout,
            shape ``[I_local, H]``.
        up_weight: Up projection in HuggingFace ``[out, in]`` layout, shape
            ``[I_local, H]``.
        down_weight: Down projection in HuggingFace ``[out, in]`` layout,
            shape ``[H, I_local]``.
        tp_group: Optional tensor-parallel process group. Gate and Up are
            column-parallel; Down is row-parallel. The platform factory uses
            deterministic fixed-tree collectives instead of backend reduction
            arithmetic. Projection shards must be equal contiguous
            intermediate-dimension ranges in process-group rank order. For
            TP=1-equivalent bit patterns, every local range must be a complete
            subtree of the canonical GEMM reduction tree.
        cp_group: Optional context-parallel process group. Each rank owns
            different token rows and the same local weight shards. Backward
            AllGathers CP tokens before running full-token ``det_gemm_db`` so
            weight gradients match CP=1 bitwise. Token shards must be equal
            contiguous flattened-token ranges in process-group rank order.
        sequence_parallel: Whether ``rmsnorm_output`` and the returned output
            are equal rank-ordered shards of the flattened token dimension
            across ``tp_group``.

    Returns:
        FFN output with shape ``[..., H]`` and dtype ``torch.bfloat16``.
    """
    _validate_ffn_inputs(rmsnorm_output, gate_weight, up_weight, down_weight)
    _require_ffn_kernels()
    if not isinstance(sequence_parallel, bool):
        raise TypeError("sequence_parallel must be a bool.")
    return _DeterministicFFNFunction.apply(
        rmsnorm_output,
        gate_weight,
        up_weight,
        down_weight,
        tp_group,
        cp_group,
        sequence_parallel,
    )
