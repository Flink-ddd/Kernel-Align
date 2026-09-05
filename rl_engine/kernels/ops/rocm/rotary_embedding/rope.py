# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Precompiled deterministic ROCm RoPE matching the shared reference layout."""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.rotary_embedding.rope import _restore_rope, _rope_table


class _RocmRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        x_2d, cos, sin = _rope_table(x, positions, theta)
        ctx.save_for_backward(cos, sin)
        ctx.x_shape = tuple(x.shape)
        ctx.pos_dim = positions.dim()
        out_2d = _C.deterministic_rope_apply_rocm(x_2d, cos, sin, 1.0)
        return _restore_rope(out_2d, x, positions)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            if ctx.pos_dim == 2 and len(ctx.x_shape) == 4:
                g_2d = grad_out.permute(1, 0, 2, 3).contiguous().reshape(-1, ctx.x_shape[-1])
                out_2d = _C.deterministic_rope_apply_rocm(g_2d, cos, sin, -1.0)
                heads, batch, seq, dim = (
                    ctx.x_shape[1],
                    ctx.x_shape[0],
                    ctx.x_shape[2],
                    ctx.x_shape[3],
                )
                grad_x = (
                    out_2d.reshape(heads, batch, seq, dim)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )
            else:
                g_2d = grad_out.contiguous().reshape(-1, grad_out.shape[-1])
                grad_x = _C.deterministic_rope_apply_rocm(g_2d, cos, sin, -1.0).reshape(
                    grad_out.shape
                )
        return grad_x, None, None


class RocmDeterministicRoPEOp:
    """Precompiled HIP RoPE path shared by ROCm training and rollout."""

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
        return self.forward(x, positions, theta=theta)

    def forward(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        if not x.is_cuda:
            raise RuntimeError("ROCm deterministic RoPE requires a GPU tensor")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("ROCm deterministic RoPE requires FP16 or BF16")
        return _RocmRoPEFunction.apply(x, positions, theta)


__all__ = ["RocmDeterministicRoPEOp"]
