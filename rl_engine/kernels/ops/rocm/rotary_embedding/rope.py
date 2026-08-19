# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic GPT-NeoX RoPE for ROCm training and inference."""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


def _build_cos_sin(
    positions: Tensor,
    half: int,
    theta: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    frequency = positions.to(device=device, dtype=torch.float32).reshape(-1, 1) * inv_freq
    return frequency.cos().contiguous(), frequency.sin().contiguous()


def _prepare(x: Tensor, positions: Tensor, theta: float) -> tuple[Tensor, Tensor, Tensor]:
    dim = x.size(-1)
    if dim % 2:
        raise ValueError(f"RoPE head_dim must be even, got {dim}")
    if positions.ndim == 1:
        table_rows = positions.numel()
        x_2d = x.contiguous().reshape(-1, dim)
    elif positions.ndim == 2:
        batch, sequence = positions.shape
        if x.ndim not in (3, 4) or x.size(0) != batch or x.size(-2) != sequence:
            raise ValueError("[B,S] positions require x [B,S,D] or [B,H,S,D]")
        x_2d = (
            x.permute(1, 0, 2, 3).contiguous().reshape(-1, dim)
            if x.ndim == 4
            else x.contiguous().reshape(-1, dim)
        )
        table_rows = batch * sequence
    else:
        raise ValueError("positions must have shape [S] or [B,S]")
    if table_rows < 1 or x_2d.size(0) % table_rows:
        raise ValueError("RoPE rows must be divisible by the position table size")
    cos, sin = _build_cos_sin(positions, dim // 2, float(theta), x.device)
    return x_2d, cos, sin


def _restore(out: Tensor, shape: torch.Size, positions_dim: int) -> Tensor:
    if positions_dim == 2 and len(shape) == 4:
        batch, heads, sequence, dim = shape
        return out.reshape(heads, batch, sequence, dim).permute(1, 0, 2, 3).contiguous()
    return out.reshape(shape)


class _RocmRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        x_2d, cos, sin = _prepare(x, positions, theta)
        ctx.save_for_backward(cos, sin)
        ctx.shape = x.shape
        ctx.positions_dim = positions.ndim
        out = _C.deterministic_rope_apply_rocm(x_2d, cos, sin, 1.0)
        return _restore(out, x.shape, positions.ndim)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        shape = ctx.shape
        if ctx.positions_dim == 2 and len(shape) == 4:
            grad_2d = grad_out.permute(1, 0, 2, 3).contiguous().reshape(-1, shape[-1])
        else:
            grad_2d = grad_out.contiguous().reshape(-1, shape[-1])
        grad_x = _C.deterministic_rope_apply_rocm(grad_2d, cos, sin, -1.0)
        return _restore(grad_x, shape, ctx.positions_dim), None, None


class RocmDeterministicRoPEOp:
    """One precompiled HIP RoPE arithmetic path shared by train and rollout."""

    backend_id = "rlkernel.rocm.deterministic_rope"
    op_class = "elementwise"
    fallback = False

    def __init__(self) -> None:
        if torch.version.hip is None:
            raise RuntimeError("RocmDeterministicRoPEOp requires a ROCm PyTorch build")
        if not _EXT_AVAILABLE or not hasattr(_C, "deterministic_rope_apply_rocm"):
            raise RuntimeError(
                "ROCm deterministic RoPE is unavailable; rebuild rl_engine._C for ROCm"
            )

    def __call__(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        if not x.is_cuda:
            raise RuntimeError("ROCm deterministic RoPE requires a GPU tensor")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("ROCm deterministic RoPE requires FP16 or BF16")
        return _RocmRoPEFunction.apply(x, positions, float(theta))


__all__ = ["RocmDeterministicRoPEOp"]
