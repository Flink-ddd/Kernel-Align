# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bias-free gated FFN backward assembled from deterministic CUDA kernels."""

from __future__ import annotations

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

    # GEMM kernels accept 2-D matrices. Flatten all leading token dimensions
    # into T while preserving H or I as the reduction/projection dimension.
    rmsnorm_output_2d = rmsnorm_output.reshape(-1, rmsnorm_output.size(-1)).contiguous()
    grad_output_2d = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
    gate_2d = gate.reshape(-1, gate.size(-1)).contiguous()
    up_2d = up.reshape(-1, up.size(-1)).contiguous()
    activated_2d = activated.reshape(-1, activated.size(-1)).contiguous()

    # Down projection: output[T,H] = activated[T,I] @ down_weight.T[I,H].
    # det_gemm_db returns activated.T @ grad_output in [I,H], then transpose it
    # back to the stored down_weight layout [H,I].
    grad_down_weight = _C.det_gemm_db(activated_2d, grad_output_2d).t().contiguous()

    # d_activated[T,I] = grad_output[T,H] @ down_weight[H,I]. The stored
    # [out,in] weight already has the exact right-hand GEMM layout, so this is a
    # direct matrix multiply rather than det_gemm_da, which would transpose B.
    grad_activated = _C.det_gemm_fwd(grad_output_2d, down_weight)

    # Differentiate activated = silu(gate) * up elementwise.
    grad_gate, grad_up = _C.swiglu_backward(grad_activated, gate_2d, up_2d)

    # Input projection weight gradients are accumulated over the T token rows.
    # det_gemm_db returns [H,I]; transpose to the stored [I,H] weight layout.
    grad_gate_weight = _C.det_gemm_db(rmsnorm_output_2d, grad_gate).t().contiguous()
    grad_up_weight = _C.det_gemm_db(rmsnorm_output_2d, grad_up).t().contiguous()

    # Both input projections consume the same RMSNorm output, so its gradient
    # is the sum of the gate and up branches. As above, [out,in] weights already
    # have the right GEMM layout: [T,I] @ [I,H] -> [T,H].
    grad_rmsnorm_output = _C.det_gemm_fwd(grad_gate, gate_weight)
    grad_rmsnorm_output.add_(_C.det_gemm_fwd(grad_up, up_weight))

    return (
        grad_rmsnorm_output.reshape_as(rmsnorm_output),
        grad_gate_weight,
        grad_up_weight,
        grad_down_weight,
    )
