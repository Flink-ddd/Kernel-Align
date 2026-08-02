# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp

_BLOCK_N = 64


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _standard_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    lse_ptr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    S_Q: tl.constexpr,
    S_KV: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_KEY_PADDING_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    kv_head = q_head // (H_Q // H_KV)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(
        q_ptr + batch * stride_qb + q_head * stride_qh + row * stride_qs + offs_d * stride_qd,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    max_score = -float("inf")
    for start_n in range(0, S_KV, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        col_mask = cols < S_KV
        k = tl.load(
            k_ptr
            + batch * stride_kb
            + kv_head * stride_kh
            + cols[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(col_mask, scores, -float("inf"))

        if CAUSAL:
            causal_keep = cols <= (row + S_KV - S_Q)
            scores = tl.where(causal_keep, scores, -float("inf"))

        if HAS_KEY_PADDING_MASK:
            keep = tl.load(mask_ptr + batch * S_KV + cols, mask=col_mask, other=0)
            scores = tl.where(keep != 0, scores, -float("inf"))

        max_score = tl.maximum(max_score, tl.max(scores, axis=0))

    denom = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start_n in range(0, S_KV, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        col_mask = cols < S_KV
        k = tl.load(
            k_ptr
            + batch * stride_kb
            + kv_head * stride_kh
            + cols[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(col_mask, scores, -float("inf"))

        if CAUSAL:
            causal_keep = cols <= (row + S_KV - S_Q)
            scores = tl.where(causal_keep, scores, -float("inf"))

        if HAS_KEY_PADDING_MASK:
            keep = tl.load(mask_ptr + batch * S_KV + cols, mask=col_mask, other=0)
            scores = tl.where(keep != 0, scores, -float("inf"))

        probs = tl.exp(scores - max_score)
        probs = tl.where(max_score == -float("inf"), 0.0, probs)
        denom += tl.sum(probs, axis=0)

        v = tl.load(
            v_ptr
            + batch * stride_vb
            + kv_head * stride_vh
            + cols[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probs[:, None] * v, axis=0)

    out = tl.where(denom > 0.0, acc / denom, 0.0)
    tl.store(
        out_ptr + batch * stride_ob + q_head * stride_oh + row * stride_os + offs_d * stride_od,
        out,
        mask=d_mask,
    )
    tl.store(lse_ptr + (batch * H_Q + q_head) * S_Q + row, max_score + tl.log(denom))


class _TritonBatchInvariantAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        causal: bool,
        scale: float,
        return_lse: bool,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.contiguous()

        batch, q_heads, q_len, head_dim = q.shape
        kv_batch, kv_heads, kv_len, k_dim = k.shape
        if v.shape != k.shape:
            raise ValueError("k and v must have the same shape")
        if kv_batch != batch:
            raise ValueError("q, k, and v must have the same batch size")
        if k_dim != head_dim:
            raise ValueError("q, k, and v must have the same head dimension")
        if q_heads % kv_heads != 0:
            raise ValueError("q heads must be divisible by k/v heads for GQA/MQA")
        if head_dim > 256:
            raise ValueError("Triton batch-invariant attention supports head_dim <= 256")
        if key_padding_mask is not None and key_padding_mask.shape != (batch, kv_len):
            raise ValueError("key_padding_mask must have shape [batch, key_seq_len]")

        out = torch.empty_like(q)
        lse = torch.empty((batch, q_heads, q_len), device=q.device, dtype=torch.float32)
        block_d = _next_power_of_2(head_dim)
        grid = (q_len, q_heads, batch)
        dummy_mask = (
            key_padding_mask
            if key_padding_mask is not None
            else q.new_empty((1,), dtype=torch.bool)
        )

        _standard_attn_fwd_kernel[grid](
            q,
            k,
            v,
            dummy_mask,
            out,
            lse,
            batch,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            scale,
            BLOCK_N=_BLOCK_N,
            BLOCK_D=block_d,
            CAUSAL=causal,
            HAS_KEY_PADDING_MASK=key_padding_mask is not None,
            num_warps=8,
        )

        ctx.save_for_backward(q, k, v, key_padding_mask)
        ctx.causal = causal
        ctx.scale = scale
        ctx.has_key_padding_mask = key_padding_mask is not None
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        q, k, v, key_padding_mask = ctx.saved_tensors
        grad_out = grad_outputs[0]
        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            out = NativeAttentionOp().forward(
                q_ref,
                k_ref,
                v_ref,
                causal=ctx.causal,
                scale=ctx.scale,
                key_padding_mask=key_padding_mask if ctx.has_key_padding_mask else None,
            )
        dq, dk, dv = torch.autograd.grad(out, (q_ref, k_ref, v_ref), grad_out)
        return dq, dk, dv, None, None, None, None


def _resolve_scale(
    head_dim: int,
    *,
    scale: Optional[float],
    softmax_scale: Optional[float],
) -> float:
    if scale is not None and softmax_scale is not None:
        raise ValueError("set only one of scale or softmax_scale")
    value = scale if scale is not None else softmax_scale
    return float(value) if value is not None else 1.0 / math.sqrt(head_dim)


def triton_batch_invariant_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scale_value = _resolve_scale(q.shape[-1], scale=scale, softmax_scale=softmax_scale)
    return _TritonBatchInvariantAttention.apply(
        q,
        k,
        v,
        key_padding_mask,
        causal,
        scale_value,
        False,
    )


def triton_batch_invariant_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    softmax_scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_value = _resolve_scale(q.shape[-1], scale=scale, softmax_scale=softmax_scale)
    return _TritonBatchInvariantAttention.apply(
        q,
        k,
        v,
        key_padding_mask,
        causal,
        scale_value,
        True,
    )


class TritonBatchInvariantAttentionOp:
    """Triton standard-softmax attention with fixed key-order reductions."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        return self.forward(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            softmax_scale=softmax_scale,
            key_padding_mask=key_padding_mask,
            dropout_p=dropout_p,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        if dropout_p != 0.0:
            raise ValueError("batch-invariant attention does not support dropout")
        return triton_batch_invariant_attention(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            softmax_scale=softmax_scale,
            key_padding_mask=key_padding_mask,
        )
