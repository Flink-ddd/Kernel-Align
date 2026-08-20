# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    SplitKVMode,
    SplitKVSpec,
)
from rl_engine.kernels.ops.cuda.attention.deterministic_attn import (
    DeterministicAttentionCoreResult,
    RLKernelDeterministicAttentionCore,
)
from rl_engine.utils.logger import logger

_MAX_TESTED_ROCM_TRITON_HEAD_DIM = 512
_AITER_API_SOURCE = "aiter.ops.mha"


class StrictRocmAttentionUnavailable(RuntimeError):
    """Raised when the exact AITER CK strict contract is unavailable."""


def _load_aiter_ck_ops() -> tuple[Callable[..., Any], Callable[..., Any], str]:
    try:
        module = importlib.import_module(_AITER_API_SOURCE)
        mha_fwd = getattr(module, "mha_fwd")
        mha_bwd = getattr(module, "mha_bwd")
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        raise StrictRocmAttentionUnavailable(
            "strict ROCm Attention requires aiter.ops.mha.mha_fwd and mha_bwd"
        ) from exc
    if not callable(mha_fwd) or not callable(mha_bwd):
        raise StrictRocmAttentionUnavailable("AITER CK MHA entry points are not callable")
    module_file = inspect.getsourcefile(module)
    if not module_file:
        raise StrictRocmAttentionUnavailable("cannot fingerprint the AITER MHA source module")
    source_sha256 = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()
    return mha_fwd, mha_bwd, source_sha256


class _AiterCKAttentionFn(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        mha_fwd: Callable[..., Any],
        mha_bwd: Callable[..., Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        result = mha_fwd(
            q_fa,
            k_fa,
            v_fa,
            0.0,
            float(scale),
            bool(causal),
            -1,
            -1,
            0,
            True,
            False,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise StrictRocmAttentionUnavailable(
                "AITER mha_fwd must return (out, lse, dropout_mask, rng_state)"
            )
        out_fa, lse, _dropout_mask, rng_state = result
        if not all(isinstance(item, torch.Tensor) for item in (out_fa, lse, rng_state)):
            raise StrictRocmAttentionUnavailable("AITER mha_fwd returned non-tensor state")
        ctx.save_for_backward(q_fa, k_fa, v_fa, out_fa, lse, rng_state)
        ctx.causal = bool(causal)
        ctx.scale = float(scale)
        ctx.mha_bwd = mha_bwd
        ctx.mark_non_differentiable(lse)
        return out_fa.transpose(1, 2).contiguous(), lse.contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor, grad_lse: torch.Tensor):
        q_fa, k_fa, v_fa, out_fa, lse, rng_state = ctx.saved_tensors
        grad_out_fa = grad_out.transpose(1, 2).contiguous()
        result = ctx.mha_bwd(
            grad_out_fa,
            q_fa,
            k_fa,
            v_fa,
            out_fa,
            lse,
            0.0,
            ctx.scale,
            ctx.causal,
            -1,
            -1,
            True,
            rng_state=rng_state,
        )
        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise StrictRocmAttentionUnavailable("AITER mha_bwd must return dQ/dK/dV")
        dq, dk, dv = result[:3]
        return (
            dq.transpose(1, 2).contiguous(),
            dk.transpose(1, 2).contiguous(),
            dv.transpose(1, 2).contiguous(),
            None,
            None,
            None,
            None,
        )


class StrictRocmAiterCKAttentionCore:
    """Shared ROCm production core using the non-Split-K AITER CK dense MHA."""

    core_id = STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_ROCM_SCHEDULE_ID
    backend_id = "aiter.rocm.ck_dense_mha"
    api_source = _AITER_API_SOURCE
    merge_order = "global_block_index"
    accum_dtype = "fp32"
    downcast_at = "final_write"
    fallback = False
    native_attention_arithmetic = True
    production_ready = True
    reference_only = False
    num_splits = 1
    split_kv_control = "dense_non_split_api"
    deterministic_backward = True

    def __init__(
        self,
        *,
        split_kv: SplitKVSpec | None = None,
        _mha_fwd: Callable[..., Any] | None = None,
        _mha_bwd: Callable[..., Any] | None = None,
        _source_sha256: str | None = None,
    ) -> None:
        requested = SplitKVSpec.disabled() if split_kv is None else split_kv
        if not isinstance(requested, SplitKVSpec):
            raise TypeError("split_kv must be a SplitKVSpec")
        if requested.mode is not SplitKVMode.DISABLED:
            raise ValueError("strict AITER CK Attention requires Split-KV to be disabled")
        if (_mha_fwd is None) != (_mha_bwd is None):
            raise ValueError("test injection requires both AITER forward and backward callables")
        if _mha_fwd is None:
            mha_fwd, mha_bwd, source_sha256 = _load_aiter_ck_ops()
        else:
            assert _mha_bwd is not None
            mha_fwd = _mha_fwd
            mha_bwd = _mha_bwd
            source_sha256 = "test-double" if _source_sha256 is None else _source_sha256
        if not callable(mha_fwd) or not callable(mha_bwd):
            raise StrictRocmAttentionUnavailable("AITER CK MHA entry points are not callable")
        self.split_kv = requested
        self.source_sha256 = source_sha256
        self._mha_fwd = mha_fwd
        self._mha_bwd = mha_bwd

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: float | None = None,
        key_padding_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
        key_position_ids: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> DeterministicAttentionCoreResult:
        self._validate_inputs(q, k, v, key_padding_mask)
        RLKernelDeterministicAttentionCore._validate_positions(
            q,
            k,
            causal=causal,
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
        )
        resolved_dtype = q.dtype if output_dtype is None else output_dtype
        if resolved_dtype != q.dtype:
            raise ValueError("strict Attention output_dtype must match the Q/K/V input dtype")
        resolved_scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else float(scale)
        out, lse = _AiterCKAttentionFn.apply(
            q,
            k,
            v,
            bool(causal),
            resolved_scale,
            self._mha_fwd,
            self._mha_bwd,
        )
        expected_lse_shape = (q.size(0), q.size(1), q.size(2))
        if out.shape != q.shape or out.dtype != resolved_dtype:
            raise StrictRocmAttentionUnavailable("AITER CK output shape/dtype changed")
        if tuple(lse.shape) != expected_lse_shape or lse.dtype != torch.float32:
            raise StrictRocmAttentionUnavailable("AITER CK must export [B,H,Sq] FP32 LSE")
        device_properties = torch.cuda.get_device_properties(q.device)
        return DeterministicAttentionCoreResult(
            out=out,
            lse=lse,
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "attention_backend": self.backend_id,
                "platform": "rocm",
                "torch_version": torch.__version__,
                "rocm_version": torch.version.hip,
                "gpu_name": device_properties.name,
                "gpu_arch": getattr(device_properties, "gcnArchName", "unknown"),
                "aiter_api_source": self.api_source,
                "aiter_source_sha256": self.source_sha256,
                "num_splits": self.num_splits,
                "split_kv_control": self.split_kv_control,
                "deterministic_backward": self.deterministic_backward,
                "dropout_p": 0.0,
                "split_kv": self.split_kv.resolve(k.size(2), backend=self.backend_id).to_dict(),
                "merge_order": self.merge_order,
                "accum_dtype": self.accum_dtype,
                "downcast_at": self.downcast_at,
                "fallback": self.fallback,
                "fallback_reason": None,
                "native_attention_arithmetic": self.native_attention_arithmetic,
                "production_ready": self.production_ready,
                "reference_only": self.reference_only,
            },
        )

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> None:
        if torch.version.hip is None:
            raise StrictRocmAttentionUnavailable("strict AITER CK core requires ROCm PyTorch")
        if key_padding_mask is not None:
            raise ValueError("strict AITER CK core materializes each unpadded logical row")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be 4-D [B,H,S,D]")
        if q.size(0) != 1 or k.size(0) != 1 or v.size(0) != 1:
            raise ValueError("strict AITER CK core executes one logical batch row at a time")
        if k.shape != v.shape or q.size(-1) != k.size(-1):
            raise ValueError("k/v shapes and q/k/v head dimensions must match")
        if q.size(1) % k.size(1) != 0:
            raise ValueError("Q heads must be divisible by KV heads for GQA")
        if q.size(-1) > 256 or q.size(-1) % 8:
            raise ValueError("AITER CK requires head_dim <= 256 and divisible by 8")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict AITER CK core supports FP16/BF16 only")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q/k/v must share one dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("strict AITER CK core requires ROCm GPU tensors")
        if not (q.device == k.device == v.device):
            raise ValueError("q/k/v must be on one ROCm device")


def _select_flash_attn_backend() -> str:
    """Select the installed FlashAttention ROCm backend."""
    return "triton"


class RocmFlashAttentionOp:
    """
    Standard FlashAttention wrapper for ROCm.
    Demonstrates the reference structure for adding new operator families.
    """

    def __init__(self):
        if torch.version.hip is None:
            raise RuntimeError("RocmFlashAttentionOp requires a ROCm PyTorch build.")

        backend = _select_flash_attn_backend()
        if backend == "triton":
            # flash-attn selects the ROCm CK/Triton backend at import time.
            os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        try:
            from flash_attn import flash_attn_func

            self.op = flash_attn_func
            logger.info("Successfully linked to external flash_attn library (%s backend).", backend)
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "ROCm FlashAttention requires a ROCm-compatible flash-attn installation. "
                "See docs/getting_started/installation.md#rocm-backend."
            ) from exc

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Standard attention forward pass.
        Args:
            q: (batch, seqlen, nheads, headdim)
            k: (batch, seqlen, nheads_k, headdim)
            v: (batch, seqlen, nheads_k, headdim)
        """
        valid_dtypes = (torch.float16, torch.bfloat16)
        if (
            q.dtype not in valid_dtypes
            or k.dtype not in valid_dtypes
            or v.dtype not in valid_dtypes
        ):
            raise TypeError("FlashAttention requires FP16 or BF16 for q/k/v")
        # PyTorch uses the CUDA device API for both CUDA and ROCm tensors.
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("Inputs must be on a CUDA/ROCm GPU device")
        if not (q.device == k.device == v.device):
            raise ValueError("q, k, and v must be on the same device")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "q, k, and v must be rank-4 tensors: (batch, seqlen, nheads, head_dim)"
            )

        head_dim = q.shape[-1]
        if head_dim == 0:
            raise ValueError("head_dim must be positive")
        if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
            raise ValueError("q, k, and v must have the same head_dim")
        if head_dim > _MAX_TESTED_ROCM_TRITON_HEAD_DIM:
            raise NotImplementedError(
                "RL-Kernel's ROCm FlashAttention wrapper currently supports "
                f"head_dim <= {_MAX_TESTED_ROCM_TRITON_HEAD_DIM}; got {head_dim}"
            )

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        return self.op(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
