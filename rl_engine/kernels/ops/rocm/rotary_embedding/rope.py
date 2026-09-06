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


class _RocmRoPEPairFunction(torch.autograd.Function):
    """Rotate Q/K with one shared position table and independent HIP launches."""

    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        theta: float,
    ) -> tuple[Tensor, Tensor]:
        ctx.set_materialize_grads(False)
        if positions.dim() != 1:
            raise ValueError("paired ROCm RoPE requires a flat position tensor")
        query_2d, cos, sin = _rope_table(query, positions, theta)
        key_2d = key.contiguous().reshape(-1, key.shape[-1])
        if key_2d.size(0) % cos.size(0):
            raise ValueError(
                f"key row count {key_2d.size(0)} is not divisible by "
                f"position count {cos.size(0)}"
            )
        ctx.save_for_backward(cos, sin)
        ctx.query_shape = tuple(query.shape)
        ctx.key_shape = tuple(key.shape)
        query_out = _C.deterministic_rope_apply_rocm(query_2d, cos, sin, 1.0)
        key_out = _C.deterministic_rope_apply_rocm(key_2d, cos, sin, 1.0)
        return query_out.reshape(query.shape), key_out.reshape(key.shape)

    @staticmethod
    def backward(ctx, grad_query: Tensor, grad_key: Tensor):
        cos, sin = ctx.saved_tensors

        def rotate_gradient(grad: Tensor | None, shape: tuple[int, ...]) -> Tensor | None:
            if grad is None:
                return None
            grad_2d = grad.contiguous().reshape(-1, shape[-1])
            return _C.deterministic_rope_apply_rocm(grad_2d, cos, sin, -1.0).reshape(shape)

        query_grad = (
            rotate_gradient(grad_query, ctx.query_shape) if ctx.needs_input_grad[0] else None
        )
        key_grad = rotate_gradient(grad_key, ctx.key_shape) if ctx.needs_input_grad[1] else None
        return query_grad, key_grad, None, None


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
        self._validate_input(x)
        return _RocmRoPEFunction.apply(x, positions, theta)

    def forward_pair(
        self,
        query: Tensor,
        key: Tensor,
        positions: Tensor,
        *,
        theta: float = 1_000_000.0,
    ) -> tuple[Tensor, Tensor]:
        """Rotate Q/K with one FP32 cos/sin table and unchanged HIP arithmetic."""

        self._validate_input(query)
        self._validate_input(key)
        if query.device != key.device or query.dtype != key.dtype:
            raise ValueError("paired ROCm RoPE Q/K must share one device and dtype")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("paired ROCm RoPE Q/K must share one head dimension")
        query_out, key_out = _RocmRoPEPairFunction.apply(query, key, positions, theta)
        # A multi-output autograd Function marks both outputs differentiable if
        # either input requires grad. Preserve the two independent-call API.
        if not query.requires_grad:
            query_out = query_out.detach()
        if not key.requires_grad:
            key_out = key_out.detach()
        return query_out, key_out

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if not x.is_cuda:
            raise RuntimeError("ROCm deterministic RoPE requires a GPU tensor")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("ROCm deterministic RoPE requires FP16 or BF16")


__all__ = ["RocmDeterministicRoPEOp"]
