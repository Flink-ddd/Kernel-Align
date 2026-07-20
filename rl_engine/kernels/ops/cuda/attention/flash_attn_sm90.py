# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

_SM90_HEAD_DIM = 128


def _sm90_hardware_available() -> bool:
    return (
        _EXT_AVAILABLE
        and hasattr(_C, "flash_attention_varlen_sm90")
        and hasattr(_C, "flash_attention_varlen_sm90_backward")
    )


def _sm90_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        _sm90_hardware_available()
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and q.dim() == 3
        and k.dim() == 3
        and v.dim() == 3
        and q.shape[-1] == _SM90_HEAD_DIM
        and k.shape[-1] == _SM90_HEAD_DIM
        and v.shape[-1] == _SM90_HEAD_DIM
        and k.shape[1] == q.shape[1]
        and v.shape[1] == q.shape[1]
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
    )


class _FlashAttentionVarlenSM90Function(torch.autograd.Function):
    """SM90 TMA+mma.sync forward and backward for packed varlen FlashAttention.

    forward() calls the compiled `_C.flash_attention_varlen_sm90` kernel and
    saves what backward's recompute-based algorithm needs. backward() calls
    `_C.flash_attention_varlen_sm90_backward`, which reuses the saved `lse`
    directly -- `p = exp(qk*scale - lse)` is algebraically identical to the
    separate-M/L form triton_attn.py's `_bwd_kernel_varlen` uses, so no
    additional forward-side state is needed.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        sm_scale,
        return_lse,
    ):
        cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
        cu_seqlens_k = cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous()

        out, lse = _C.flash_attention_varlen_sm90(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            bool(causal),
            float(sm_scale),
        )

        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_k = int(max_seqlen_k)
        ctx.causal = bool(causal)
        ctx.sm_scale = float(sm_scale)

        if not return_lse:
            return out, None
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, do, _dlse):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        do = do.contiguous()

        dq, dk, dv = _C.flash_attention_varlen_sm90_backward(
            do,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_k,
            ctx.causal,
            ctx.sm_scale,
        )
        # Inputs: q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        # max_seqlen_k, causal, sm_scale, return_lse.
        return dq, dk, dv, None, None, None, None, None, None, None


class FlashAttentionVarlenSM90Op:
    """SM90 (Hopper) packed variable-length FlashAttention, forward + backward.

    TMA + `mma.sync` causal attention with attention-domain LSE export
    (M + log(L), not the vocab-domain LSE used by the fused_logp /
    linear_logp kernels), matching the semantics of
    `triton_flash_attention_varlen` -- the cross-platform baseline this
    kernel is validated against.

    Requires the extension built with `KERNEL_ALIGN_FORCE_SM90=1` on a Hopper
    (SM90) device; bfloat16 q/k/v, head_dim=128, contiguous, no GQA only, and
    *both* the forward and backward native symbols present -- a build with
    only the forward kernel compiled falls back to Triton entirely rather
    than mixing an SM90 forward with a Triton backward. Falls back to
    `triton_flash_attention_varlen` (itself a proper autograd Function, so
    gradients still work through the fallback) for anything outside that
    support surface.
    """

    def __init__(self) -> None:
        self.has_hardware_op = _sm90_hardware_available()
        if self.has_hardware_op:
            logger.info(
                "Successfully linked to RL-Kernel _C.flash_attention_varlen_sm90"
                " (+ backward)."
            )
        else:
            logger.warning(
                "RL-Kernel _C.flash_attention_varlen_sm90[_backward] is unavailable. "
                "FlashAttentionVarlenSM90Op will fall back to triton_flash_attention_varlen "
                "(rebuild with KERNEL_ALIGN_FORCE_SM90=1 on a Hopper GPU for the fused kernel)."
            )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = True,
        sm_scale: float | None = None,
        return_lse: bool = False,
    ):
        if sm_scale is None:
            sm_scale = 1.0 / (q.shape[-1] ** 0.5)

        if self.has_hardware_op and _sm90_supported(q, k, v):
            out, lse = _FlashAttentionVarlenSM90Function.apply(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal,
                sm_scale,
                return_lse,
            )
            return (out, lse) if return_lse else out

        from rl_engine.kernels.ops.triton.triton_attn import triton_flash_attention_varlen

        return triton_flash_attention_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=causal,
            sm_scale=sm_scale,
            return_lse=return_lse,
        )


_OP: FlashAttentionVarlenSM90Op | None = None


def _get_op() -> FlashAttentionVarlenSM90Op:
    global _OP
    if _OP is None:
        _OP = FlashAttentionVarlenSM90Op()
    return _OP


def flash_attention_sm90_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = True,
    sm_scale: float | None = None,
    return_lse: bool = False,
):
    """SM90 packed variable-length FlashAttention (forward + backward).

    Signature mirrors `triton_flash_attention_varlen`
    (`rl_engine.kernels.ops.triton.triton_attn`) exactly, so call sites and
    tests can swap between the two backends with no changes. Differentiable:
    `q`/`k`/`v` gradients flow correctly whether the SM90 hardware kernel or
    the Triton fallback services the call.

    Args:
        q: [total_q, H, D] packed queries. D must be 128 for the fused SM90
            kernel to engage; other shapes/dtypes fall back to Triton.
        k, v: [total_k, H, D] packed keys/values. GQA (Hk != Hq) unsupported.
        cu_seqlens_q, cu_seqlens_k: int32 [batch + 1] cumulative offsets,
            cu_seqlens[0] == 0.
        max_seqlen_q, max_seqlen_k: max per-sequence length in the batch.
        causal: causal masking, anchored per-sequence via `Skv - Sq`.
        sm_scale: defaults to `1/sqrt(D)`.
        return_lse: if True, also return the packed `[total_q, H]` float32
            attention-domain LSE (non-differentiable).
    Returns:
        `out` of shape [total_q, H, D] if `return_lse` is False, else
        `(out, lse)`.
    """
    return _get_op()(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
        sm_scale=sm_scale,
        return_lse=return_lse,
    )
