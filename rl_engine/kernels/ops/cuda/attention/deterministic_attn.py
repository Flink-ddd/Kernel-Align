# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA deterministic standard-softmax attention (issue #147).

Forward: QK → masked softmax+LSE → PV (all FP32 intermediate).
Backward: dP → softmax_bwd → dQ/dK/dV with §4.1 fixed GQA order.
Wrapped in autograd.Function so #108 harness can .backward() through it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

_HEAD_DIM = 128


class _DeterministicAttentionFn(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        key_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        mask_c = key_padding_mask.contiguous() if key_padding_mask is not None else None

        results = _C.deterministic_attention_forward(q_c, k_c, v_c, causal, float(scale), mask_c)
        out, lse, P = results[0], results[1], results[2]

        ctx.save_for_backward(q_c, k_c, v_c, P, mask_c)
        ctx.causal = causal
        ctx.scale = scale
        ctx.mark_non_differentiable(lse)

        return out, lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor, grad_lse: torch.Tensor):
        q_c, k_c, v_c, P, mask_c = ctx.saved_tensors

        dQ, dK, dV = _C.deterministic_attention_backward(
            grad_out.contiguous(),
            q_c,
            k_c,
            v_c,
            P,
            ctx.causal,
            float(ctx.scale),
            mask_c,
        )

        return dQ, dK, dV, None, None, None


class DeterministicAttentionOp:
    """Batch-invariant standard softmax attention on CUDA.

    Materializes full FP32 scores/P. Public surface matches NativeAttentionOp
    so #108 harness can call forward(**inputs) with key_padding_mask.
    """

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "deterministic_attention_forward"):
            raise RuntimeError(
                "Deterministic CUDA attention kernel is unavailable. "
                "Rebuild the extension with `pip install -e .` on a CUDA build."
            )
        if not hasattr(_C, "deterministic_attention_backward"):
            raise RuntimeError(
                "Deterministic CUDA attention backward kernel is unavailable. "
                "Rebuild the extension with `pip install -e .` on a CUDA build."
            )
        logger.info("Successfully linked to _C.deterministic_attention_forward/backward.")

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Harness / registry main path: return out only. Differentiable."""
        out, _lse = self.forward_with_lse(
            q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask
        )
        return out

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (out, lse) with FP32 LSE for debug / handoff hooks."""
        self._validate_inputs(q, k, v, key_padding_mask)
        resolved_scale = scale if scale is not None else (1.0 / math.sqrt(q.shape[-1]))
        out, lse = _DeterministicAttentionFn.apply(
            q, k, v, causal, resolved_scale, key_padding_mask
        )
        return out, lse

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> None:
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"q/k/v must be 4-D [B, H, S, D], got q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        b, hq, sq, d = q.shape
        hkv, skv = k.shape[1], k.shape[2]
        if k.shape[0] != b or v.shape[0] != b:
            raise ValueError("batch size mismatch between q/k/v")
        if v.shape[1] != hkv or v.shape[2] != skv or k.shape[3] != d or v.shape[3] != d:
            raise ValueError(
                f"k/v shape mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}, "
                f"expected k/v [B={b}, Hkv, Skv, D={d}]"
            )
        if d != _HEAD_DIM:
            raise ValueError(f"head dim D must be {_HEAD_DIM}, got {d}")
        if hq % hkv != 0:
            raise ValueError(f"Hq={hq} not divisible by Hkv={hkv} (GQA group)")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"only FP16/BF16 supported, got {q.dtype}")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q, k, v must share the same dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("q, k, v must be CUDA tensors")
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")
            if key_padding_mask.shape != (b, skv):
                raise ValueError(
                    f"key_padding_mask must be [B, Skv]=[{b}, {skv}], "
                    f"got {tuple(key_padding_mask.shape)}"
                )
        if sq < 1 or skv < 1:
            raise ValueError(f"Sq and Skv must be positive, got Sq={sq}, Skv={skv}")
