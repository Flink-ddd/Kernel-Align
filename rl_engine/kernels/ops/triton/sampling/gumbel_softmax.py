# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.pytorch.sampling.gumbel_softmax import (
    _validate_gumbel_softmax_inputs,
)

_BLOCK_V = 1024


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
    row_off = row.to(tl.int64) * V

    z_max = -float("inf")
    for start in range(0, V, BLOCK_V):
        offs = start + cols
        mask = offs < V
        logits = tl.load(logits_ptr + row_off + offs, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        if HAS_GUMBELS:
            gumbels = tl.load(gumbels_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            u = tl.rand(seed, row_off + offs)
            u = tl.minimum(tl.maximum(u, 1.0e-20), 1.0 - 1.0e-7)
            gumbels = -tl.log(-tl.log(u))
        z = tl.where(mask, (logits + gumbels) / tau, -float("inf"))
        z_max = tl.maximum(z_max, tl.max(z, axis=0))

    denom = 0.0
    for start in range(0, V, BLOCK_V):
        offs = start + cols
        mask = offs < V
        logits = tl.load(logits_ptr + row_off + offs, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        if HAS_GUMBELS:
            gumbels = tl.load(gumbels_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            u = tl.rand(seed, row_off + offs)
            u = tl.minimum(tl.maximum(u, 1.0e-20), 1.0 - 1.0e-7)
            gumbels = -tl.log(-tl.log(u))
        z = tl.where(mask, (logits + gumbels) / tau, -float("inf"))
        denom += tl.sum(tl.exp(z - z_max), axis=0)

    for start in range(0, V, BLOCK_V):
        offs = start + cols
        mask = offs < V
        logits = tl.load(logits_ptr + row_off + offs, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        if HAS_GUMBELS:
            gumbels = tl.load(gumbels_ptr + row_off + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            u = tl.rand(seed, row_off + offs)
            u = tl.minimum(tl.maximum(u, 1.0e-20), 1.0 - 1.0e-7)
            gumbels = -tl.log(-tl.log(u))
        z = tl.where(mask, (logits + gumbels) / tau, -float("inf"))
        y = tl.exp(z - z_max) / denom
        y = tl.where(mask, y, 0.0)
        tl.store(y_soft_ptr + row_off + offs, y, mask=mask)
        if HARD:
            out = tl.where(z == z_max, 1.0, 0.0)
            tl.store(out_ptr + row_off + offs, out, mask=mask)
        else:
            tl.store(out_ptr + row_off + offs, y, mask=mask)


class _GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, gumbels, tau: float, hard: bool, seed: int):
        V = logits.shape[-1]
        lead_shape = logits.shape[:-1]
        logits_2d = logits.contiguous().view(-1, V)
        gumbels_2d = (
            gumbels.contiguous().view(-1, V) if gumbels is not None else logits_2d
        )
        n_rows = logits_2d.shape[0]
        block_v = min(_BLOCK_V, triton.next_power_of_2(V))

        y_soft = torch.empty_like(logits_2d, dtype=torch.float32)
        out = torch.empty_like(logits_2d, dtype=torch.float32)
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
        )

        ctx.save_for_backward(y_soft)
        ctx.tau = float(tau)
        ctx.logits_shape = tuple(logits.shape)
        ctx.logits_dtype = logits.dtype
        return out.view(*lead_shape, V).to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (y_soft,) = ctx.saved_tensors
        grad_2d = grad_output.contiguous().view_as(y_soft).to(torch.float32)
        dot = (grad_2d * y_soft).sum(dim=-1, keepdim=True)
        grad_logits = y_soft * (grad_2d - dot) / ctx.tau
        return grad_logits.view(ctx.logits_shape).to(ctx.logits_dtype), None, None, None, None


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
        if logits.device.type not in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                "TritonGumbelSoftmaxOp requires a GPU tensor (CUDA / ROCm / XPU), got "
                f"device '{logits.device}'."
            )
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (), device="cpu").item())
        return _GumbelSoftmaxFunction.apply(logits, gumbels, float(tau), bool(hard), int(seed))
