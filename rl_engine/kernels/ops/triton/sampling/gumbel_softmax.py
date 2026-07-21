# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.pytorch.sampling.gumbel_softmax import (
    NativeGumbelSoftmaxOp,
    _validate_gumbel_softmax_inputs,
)

_MAX_BLOCK_V = 131072


def _launch_config(vocab_size: int) -> tuple[int, int]:
    block_v = triton.next_power_of_2(vocab_size)
    if block_v > _MAX_BLOCK_V:
        raise ValueError(
            f"vocab size {vocab_size} exceeds TritonGumbelSoftmaxOp limit {_MAX_BLOCK_V}"
        )
    num_warps = 32 if block_v >= 65536 else 16 if block_v >= 32768 else 8
    return block_v, num_warps


@triton.jit
def _gumbel_softmax_fwd_kernel(
    logits_ptr,
    gumbels_ptr,
    y_soft_ptr,
    out_ptr,
    seed,
    n_rows,
    V: tl.constexpr,
    tau: tl.constexpr,
    HAS_GUMBELS: tl.constexpr,
    HARD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_V)
    mask = cols < V
    row_off = row.to(tl.int64) * V

    logits = tl.load(logits_ptr + row_off + cols, mask=mask, other=-float("inf")).to(tl.float32)
    if HAS_GUMBELS:
        gumbels = tl.load(gumbels_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        u = tl.rand(seed, row_off + cols)
        u = tl.minimum(tl.maximum(u, 1.0e-20), 1.0 - 1.0e-7)
        gumbels = -tl.log(-tl.log(u))

    z = tl.where(mask, (logits + gumbels) / tau, -float("inf"))
    z_max = tl.max(z, axis=0)
    exp_z = tl.exp(z - z_max)
    denom = tl.sum(exp_z, axis=0)
    y = tl.where(mask, exp_z / denom, 0.0)

    tl.store(y_soft_ptr + row_off + cols, y, mask=mask)
    if HARD:
        # Tie-break to the first maximum so each row is exactly one-hot.
        is_max = z == z_max
        first_max = tl.min(tl.where(is_max, cols, BLOCK_V), axis=0)
        out = tl.where(cols == first_max, 1.0, 0.0)
        tl.store(out_ptr + row_off + cols, out, mask=mask)
    else:
        tl.store(out_ptr + row_off + cols, y, mask=mask)


@triton.jit
def _gumbel_softmax_hard_nograd_kernel(
    logits_ptr,
    gumbels_ptr,
    out_ptr,
    seed,
    n_rows,
    V: tl.constexpr,
    HAS_GUMBELS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_V)
    mask = cols < V
    row_off = row.to(tl.int64) * V

    logits = tl.load(logits_ptr + row_off + cols, mask=mask, other=-float("inf")).to(tl.float32)
    if HAS_GUMBELS:
        gumbels = tl.load(gumbels_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    else:
        u = tl.rand(seed, row_off + cols)
        u = tl.minimum(tl.maximum(u, 1.0e-20), 1.0 - 1.0e-7)
        gumbels = -tl.log(-tl.log(u))

    z = tl.where(mask, logits + gumbels, -float("inf"))
    z_max = tl.max(z, axis=0)
    is_max = z == z_max
    first_max = tl.min(tl.where(is_max, cols, BLOCK_V), axis=0)
    out = tl.where(cols == first_max, 1.0, 0.0)
    tl.store(out_ptr + row_off + cols, out, mask=mask)


class _GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, gumbels, tau: float, hard: bool, seed: int):
        V = logits.shape[-1]
        lead_shape = logits.shape[:-1]
        logits_2d = logits.contiguous().view(-1, V)
        gumbels_2d = gumbels.contiguous().view(-1, V) if gumbels is not None else logits_2d
        n_rows = logits_2d.shape[0]
        block_v, num_warps = _launch_config(V)

        y_soft = torch.empty_like(logits_2d)
        out = torch.empty_like(logits_2d) if hard else y_soft
        _gumbel_softmax_fwd_kernel[(n_rows,)](
            logits_2d,
            gumbels_2d,
            y_soft,
            out,
            int(seed),
            n_rows,
            V,
            float(tau),
            HAS_GUMBELS=gumbels is not None,
            HARD=bool(hard),
            BLOCK_V=block_v,
            num_warps=num_warps,
        )

        ctx.save_for_backward(y_soft)
        ctx.tau = float(tau)
        ctx.logits_shape = tuple(logits.shape)
        ctx.logits_dtype = logits.dtype
        return out.view(*lead_shape, V)

    @staticmethod
    def backward(ctx, grad_output):
        (y_soft,) = ctx.saved_tensors
        grad_2d = grad_output.contiguous().view_as(y_soft)
        grad_logits = torch._softmax_backward_data(
            grad_2d,
            y_soft,
            -1,
            ctx.logits_dtype,
        )
        if ctx.tau != 1.0:
            grad_logits = grad_logits / ctx.tau
        return grad_logits.view(ctx.logits_shape).to(ctx.logits_dtype), None, None, None, None


def _hard_gumbel_softmax_nograd(
    logits: torch.Tensor,
    gumbels: Optional[torch.Tensor],
    seed: int,
) -> torch.Tensor:
    V = logits.shape[-1]
    lead_shape = logits.shape[:-1]
    logits_2d = logits.contiguous().view(-1, V)
    gumbels_2d = gumbels.contiguous().view(-1, V) if gumbels is not None else logits_2d
    block_v, num_warps = _launch_config(V)

    out = torch.empty_like(logits_2d)
    _gumbel_softmax_hard_nograd_kernel[(logits_2d.shape[0],)](
        logits_2d,
        gumbels_2d,
        out,
        int(seed),
        logits_2d.shape[0],
        V,
        HAS_GUMBELS=gumbels is not None,
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
    return out.view(*lead_shape, V)


class TritonGumbelSoftmaxOp:
    """Triton Gumbel-Softmax sampler with straight-through hard samples."""

    def __call__(
        self,
        logits: torch.Tensor,
        *,
        tau: float = 1.0,
        hard: bool = False,
        gumbels: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        return self.apply(logits, tau=tau, hard=hard, gumbels=gumbels, seed=seed)

    def apply(
        self,
        logits: torch.Tensor,
        *,
        tau: float = 1.0,
        hard: bool = False,
        gumbels: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        _validate_gumbel_softmax_inputs(logits, float(tau), gumbels)
        if triton.next_power_of_2(logits.shape[-1]) > _MAX_BLOCK_V:
            return NativeGumbelSoftmaxOp()(logits, tau=tau, hard=hard, gumbels=gumbels)
        if logits.device.type not in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                "TritonGumbelSoftmaxOp requires a GPU tensor (CUDA / ROCm / XPU), got "
                f"device '{logits.device}'."
            )
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (), device="cpu").item())
        if hard and (not torch.is_grad_enabled() or not logits.requires_grad):
            return _hard_gumbel_softmax_nograd(logits, gumbels, int(seed))
        return _GumbelSoftmaxFunction.apply(logits, gumbels, float(tau), bool(hard), int(seed))
