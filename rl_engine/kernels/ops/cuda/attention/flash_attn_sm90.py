# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

_SM90_HEAD_DIM = 128


class FlashAttentionVarlenSM90Op:
    """SM90 (Hopper) forward-only packed variable-length FlashAttention.

    TMA + `mma.sync` causal attention with attention-domain LSE export
    (M + log(L), not the vocab-domain LSE used by the fused_logp /
    linear_logp kernels), matching the semantics of
    `triton_flash_attention_varlen` -- the cross-platform baseline this
    kernel is validated against. Forward only this milestone: no autograd,
    no backward.

    Requires the extension built with `KERNEL_ALIGN_FORCE_SM90=1` on a Hopper
    (SM90) device; bfloat16 q/k/v, head_dim=128, contiguous, no GQA only.
    Falls back to `triton_flash_attention_varlen` (not a hand-rolled SDPA
    fallback) for anything outside that support surface, since the Triton
    path is the exact semantic reference, not a third implementation.
    """

    def __init__(self) -> None:
        self.has_hardware_op = _EXT_AVAILABLE and hasattr(_C, "flash_attention_varlen_sm90")
        if self.has_hardware_op:
            self.op = _C.flash_attention_varlen_sm90
            logger.info("Successfully linked to RL-Kernel _C.flash_attention_varlen_sm90.")
        else:
            logger.warning(
                "RL-Kernel _C.flash_attention_varlen_sm90 is unavailable. "
                "FlashAttentionVarlenSM90Op will fall back to triton_flash_attention_varlen "
                "(rebuild with KERNEL_ALIGN_FORCE_SM90=1 on a Hopper GPU for the fused kernel)."
            )

    def _supported(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
        return (
            self.has_hardware_op
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

        if self._supported(q, k, v):
            cu_seqlens_q_i32 = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
            cu_seqlens_k_i32 = cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous()
            out, lse = self.op(
                q,
                k,
                v,
                cu_seqlens_q_i32,
                cu_seqlens_k_i32,
                int(max_seqlen_q),
                int(max_seqlen_k),
                bool(causal),
                float(sm_scale),
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
    """SM90 forward-only packed variable-length FlashAttention.

    Signature mirrors `triton_flash_attention_varlen`
    (`rl_engine.kernels.ops.triton.triton_attn`) exactly, so call sites and
    tests can swap between the two backends with no changes.

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
            attention-domain LSE.
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
