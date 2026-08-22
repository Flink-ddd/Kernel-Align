# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic Triton embedding with an atomic-free backward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.backward_runtime import record_backward


@triton.jit
def _embedding_fwd(ids, weight, out, n_tokens, hidden: tl.constexpr, block_h: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, block_h)
    token = tl.load(ids + row)
    values = tl.load(weight + token * hidden + offs, mask=offs < hidden, other=0.0)
    tl.store(out + row * hidden + offs, values, mask=offs < hidden)


@triton.jit
def _embedding_bwd(
    ids,
    grad_rows,
    grad_weight,
    n_tokens: tl.constexpr,
    hidden: tl.constexpr,
    block_t: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1)
    offs = tl.arange(0, block_t)
    acc = tl.zeros((), tl.float32)
    for start in range(0, n_tokens, block_t):
        rows = start + offs
        mask = rows < n_tokens
        row_ids = tl.load(ids + rows, mask=mask, other=-1)
        values = tl.load(grad_rows + rows * hidden + col, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(row_ids == token, values, 0.0), axis=0)
    tl.store(grad_weight + token * hidden + col, acc)


class _TritonEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ids = token_ids.reshape(-1).to(dtype=torch.int64).contiguous()
        vocab, hidden = weight.shape
        if ids.numel() and bool(((ids < 0) | (ids >= vocab)).any()):
            raise ValueError(f"token_ids must be in [0, {vocab})")
        out = torch.empty((ids.numel(), hidden), device=weight.device, dtype=weight.dtype)
        _embedding_fwd[(ids.numel(),)](
            ids,
            weight.contiguous(),
            out,
            ids.numel(),
            hidden=hidden,
            block_h=triton.next_power_of_2(hidden),
        )
        ctx.save_for_backward(ids)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.output_shape = tuple(token_ids.shape) + (hidden,)
        return out.reshape(ctx.output_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (ids,) = ctx.saved_tensors
        vocab, hidden = ctx.weight_shape
        grad_rows = grad_output.reshape(-1, hidden).contiguous()
        grad_weight = torch.empty(
            (vocab, hidden), device=grad_output.device, dtype=ctx.weight_dtype
        )
        _embedding_bwd[(vocab, hidden)](
            ids,
            grad_rows,
            grad_weight,
            n_tokens=ids.numel(),
            hidden=hidden,
            block_t=64,
        )
        record_backward(
            "embedding",
            kernel_id="rl_engine.kernels.ops.triton.linear.embedding._embedding_bwd",
            impl="triton_embedding_bwd",
            family="triton",
        )
        return None, grad_weight


class TritonEmbeddingOp:
    """Table lookup with one program per row and deterministic weight VJP."""

    op_class = "elementwise"
    is_batch_invariant = True

    def __call__(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids, weight)

    def forward(self, token_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not token_ids.is_cuda or not weight.is_cuda:
            raise RuntimeError("TritonEmbeddingOp requires CUDA tensors")
        return _TritonEmbeddingFunction.apply(token_ids, weight)
