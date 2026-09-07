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
    SplitKVExecutionPlan,
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
_AITER_OP_NAMESPACE = "aiter"

# AITER wraps its kernels in a JIT loader whose Python signature is
# ``(*args, **kwargs)``, so ``inspect.signature`` cannot see the contract the
# way it can for the FA4 CuTe API. The names live in the registered Torch
# schema instead, and the calls below are positional, so the order is part of
# what has to hold: an upstream insertion would silently reinterpret every
# argument after it. These tuples are the exact positional prefix each call
# site assumes.
_AITER_FWD_POSITIONAL_CONTRACT = (
    "q",
    "k",
    "v",
    "dropout_p",
    "softmax_scale",
    "is_causal",
    "window_size_left",
    "window_size_right",
    "sink_size",
    "return_softmax_lse",
    "return_dropout_randval",
)
_AITER_FWD_REQUIRED_KEYWORDS = frozenset({"out"})
_AITER_BWD_POSITIONAL_CONTRACT = (
    "dout",
    "q",
    "k",
    "v",
    "out",
    "softmax_lse",
    "dropout_p",
    "softmax_scale",
    "is_causal",
    "window_size_left",
    "window_size_right",
    "deterministic",
)
# Passed by keyword, so only presence matters.
_AITER_BWD_REQUIRED_KEYWORDS = frozenset({"rng_state"})

# Stable dispatch identity for the strict ROCm attention core.  Kept at module
# scope so contract-aware dispatch and the Vime adapter name one constant
# instead of duplicating the string.
BACKEND_ID = "aiter.rocm.ck_dense_mha"


class StrictRocmAttentionUnavailable(RuntimeError):
    """Raised when the exact AITER CK strict contract is unavailable."""


def _aiter_schema_argument_names(op_name: str) -> tuple[str, ...]:
    """Return the registered Torch schema argument names for one AITER op."""

    namespace = getattr(torch.ops, _AITER_OP_NAMESPACE, None)
    if namespace is None:
        raise StrictRocmAttentionUnavailable(
            f"the '{_AITER_OP_NAMESPACE}' Torch operator namespace is not registered"
        )
    try:
        overload = getattr(namespace, op_name).default
        arguments = overload._schema.arguments
    except (AttributeError, RuntimeError) as exc:
        raise StrictRocmAttentionUnavailable(
            f"cannot read the Torch schema for {_AITER_OP_NAMESPACE}::{op_name}"
        ) from exc
    return tuple(argument.name for argument in arguments)


def _validate_aiter_schema(
    op_name: str,
    positional_contract: tuple[str, ...],
    *,
    required_keywords: frozenset[str] = frozenset(),
) -> None:
    """Fail closed unless AITER still accepts what the call sites pass.

    The strict calls are positional, so a renamed *or reordered* argument
    changes their meaning without changing their shape. Checking the ordered
    prefix catches both, which name-presence alone would not.
    """

    names = _aiter_schema_argument_names(op_name)
    prefix = names[: len(positional_contract)]
    if prefix != positional_contract:
        raise StrictRocmAttentionUnavailable(
            f"AITER {op_name} positional contract changed: strict ROCm Attention "
            f"passes {positional_contract} but the schema declares {prefix}"
        )
    missing = sorted(required_keywords.difference(names))
    if missing:
        raise StrictRocmAttentionUnavailable(
            f"AITER {op_name} is missing strict controls: " + ", ".join(missing)
        )


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
    _validate_aiter_schema(
        "mha_fwd",
        _AITER_FWD_POSITIONAL_CONTRACT,
        required_keywords=_AITER_FWD_REQUIRED_KEYWORDS,
    )
    _validate_aiter_schema(
        "mha_bwd",
        _AITER_BWD_POSITIONAL_CONTRACT,
        required_keywords=_AITER_BWD_REQUIRED_KEYWORDS,
    )
    module_file = inspect.getsourcefile(module)
    if not module_file:
        raise StrictRocmAttentionUnavailable("cannot fingerprint the AITER MHA source module")
    source_sha256 = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()
    return mha_fwd, mha_bwd, source_sha256


def _call_aiter_mha_fwd_into(
    mha_fwd: Callable[..., Any],
    q_fa: torch.Tensor,
    k_fa: torch.Tensor,
    v_fa: torch.Tensor,
    *,
    causal: bool,
    scale: float,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        out=out,
    )
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise StrictRocmAttentionUnavailable(
            "AITER mha_fwd must return (out, lse, dropout_mask, rng_state)"
        )
    out_fa, lse, _dropout_mask, rng_state = result
    if not all(isinstance(item, torch.Tensor) for item in (out_fa, lse, rng_state)):
        raise StrictRocmAttentionUnavailable("AITER mha_fwd returned non-tensor state")
    return out_fa, lse, rng_state


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
    backend_id = BACKEND_ID
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
        # Decode invokes this core once per KV group. Keep only the last
        # immutable lookup result so every new sequence length still resolves
        # once, while remaining same-length groups/layers avoid duplicate host
        # work.
        self._device_description_cache: tuple[torch.device, tuple[str, str]] | None = None
        self._split_kv_plan_cache: (
            tuple[tuple[SplitKVSpec, int, str], SplitKVExecutionPlan] | None
        ) = None

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
        result_out, lse = _AiterCKAttentionFn.apply(
            q,
            k,
            v,
            bool(causal),
            resolved_scale,
            self._mha_fwd,
            self._mha_bwd,
        )
        return self._finish_result(q, k, result_out, lse, resolved_dtype)

    def forward_decode_with_lse_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: torch.Tensor,
        scale: float | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> DeterministicAttentionCoreResult:
        """Run single-token no-grad decode directly into a caller buffer."""

        return self._forward_decode_with_lse_into(
            q,
            k,
            v,
            out=out,
            scale=scale,
            output_dtype=output_dtype,
            logical_group_batch=False,
        )

    def forward_grouped_decode_with_lse_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: torch.Tensor,
        scale: float | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> DeterministicAttentionCoreResult:
        """Run independent one-KV-group decode rows in one AITER launch.

        Every AITER batch row still contains exactly one logical KV group. The
        batching only removes duplicate host dispatches; it does not combine
        heads or alter the per-row reduction tree.
        """

        return self._forward_decode_with_lse_into(
            q,
            k,
            v,
            out=out,
            scale=scale,
            output_dtype=output_dtype,
            logical_group_batch=True,
        )

    def _forward_decode_with_lse_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: torch.Tensor,
        scale: float | None,
        output_dtype: torch.dtype | None,
        logical_group_batch: bool,
    ) -> DeterministicAttentionCoreResult:
        """Shared direct-output implementation for scalar and grouped decode."""

        if logical_group_batch:
            self._validate_inputs(
                q,
                k,
                v,
                None,
                allow_logical_group_batch=True,
            )
        else:
            self._validate_inputs(q, k, v, None)
        self._validate_direct_output(out, q, k, v)
        resolved_dtype = q.dtype if output_dtype is None else output_dtype
        if resolved_dtype != q.dtype:
            raise ValueError("strict Attention output_dtype must match the Q/K/V input dtype")
        resolved_scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else float(scale)
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        out_fa = out.transpose(1, 2)
        returned_out, lse, _rng_state = _call_aiter_mha_fwd_into(
            self._mha_fwd,
            q_fa,
            k_fa,
            v_fa,
            causal=False,
            scale=resolved_scale,
            out=out_fa,
        )
        if (
            returned_out.data_ptr() != out_fa.data_ptr()
            or returned_out.shape != out_fa.shape
            or returned_out.dtype != out_fa.dtype
            or returned_out.device != out_fa.device
        ):
            raise StrictRocmAttentionUnavailable(
                "AITER CK did not write to the requested output buffer"
            )
        return self._finish_result(
            q,
            k,
            out,
            lse.contiguous(),
            resolved_dtype,
            extra_provenance={
                "output_buffer_reused": True,
                "core_output_staging": (
                    "aiter_direct_caller_group_batch"
                    if logical_group_batch
                    else "aiter_direct_caller_group"
                ),
                "logical_group_batch_size": q.size(0),
            },
        )

    def _finish_result(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        resolved_dtype: torch.dtype,
        *,
        extra_provenance: dict[str, object] | None = None,
    ) -> DeterministicAttentionCoreResult:
        expected_lse_shape = (q.size(0), q.size(1), q.size(2))
        if out.shape != q.shape or out.dtype != resolved_dtype:
            raise StrictRocmAttentionUnavailable("AITER CK output shape/dtype changed")
        if tuple(lse.shape) != expected_lse_shape or lse.dtype != torch.float32:
            raise StrictRocmAttentionUnavailable("AITER CK must export [B,H,Sq] FP32 LSE")
        gpu_name, gpu_arch = self._device_description(q.device)
        split_kv_plan = self._resolve_split_kv_plan(k.size(2))
        provenance: dict[str, object] = {
            "strict_core_id": self.core_id,
            "strict_schedule": self.strict_schedule,
            "attention_backend": self.backend_id,
            "platform": "rocm",
            "torch_version": torch.__version__,
            "rocm_version": torch.version.hip,
            "gpu_name": gpu_name,
            "gpu_arch": gpu_arch,
            "aiter_api_source": self.api_source,
            "aiter_source_sha256": self.source_sha256,
            "num_splits": self.num_splits,
            "split_kv_control": self.split_kv_control,
            "deterministic_backward": self.deterministic_backward,
            "dropout_p": 0.0,
            # to_dict deliberately remains per result: callers receive fresh
            # mutable dictionaries/lists even though the frozen plan is reused.
            "split_kv": split_kv_plan.to_dict(),
            "merge_order": self.merge_order,
            "accum_dtype": self.accum_dtype,
            "downcast_at": self.downcast_at,
            "fallback": self.fallback,
            "fallback_reason": None,
            "native_attention_arithmetic": self.native_attention_arithmetic,
            "production_ready": self.production_ready,
            "reference_only": self.reference_only,
        }
        if extra_provenance is not None:
            provenance.update(extra_provenance)
        return DeterministicAttentionCoreResult(
            out=out,
            lse=lse,
            provenance=provenance,
        )

    def _device_description(self, device: torch.device) -> tuple[str, str]:
        cached = self._device_description_cache
        if cached is not None and cached[0] == device:
            return cached[1]
        properties = torch.cuda.get_device_properties(device)
        description = (properties.name, getattr(properties, "gcnArchName", "unknown"))
        self._device_description_cache = (device, description)
        return description

    def _resolve_split_kv_plan(self, total_kv_tokens: int) -> SplitKVExecutionPlan:
        key = (self.split_kv, total_kv_tokens, self.backend_id)
        cached = self._split_kv_plan_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        plan = self.split_kv.resolve(total_kv_tokens, backend=self.backend_id)
        self._split_kv_plan_cache = (key, plan)
        return plan

    @staticmethod
    def _validate_direct_output(
        out: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if torch.is_grad_enabled():
            raise ValueError("AITER CK direct output requires disabled gradient mode")
        if q.size(2) != 1:
            raise ValueError("AITER CK direct output is restricted to single-token decode")
        if k.size(1) != 1:
            raise ValueError("AITER CK direct output requires exactly one KV group")
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError("AITER CK direct output must match the Q shape, dtype, and device")
        if not out.is_contiguous():
            raise ValueError("AITER CK direct output must be contiguous")
        if out.requires_grad or any(tensor.requires_grad for tensor in (q, k, v)):
            raise ValueError("AITER CK direct output is restricted to no-grad decode")
        output_storage = out.untyped_storage().data_ptr()
        if any(tensor.untyped_storage().data_ptr() == output_storage for tensor in (q, k, v)):
            raise ValueError("AITER CK direct output must not alias Q/K/V storage")

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        *,
        allow_logical_group_batch: bool = False,
    ) -> None:
        if torch.version.hip is None:
            raise StrictRocmAttentionUnavailable("strict AITER CK core requires ROCm PyTorch")
        if key_padding_mask is not None:
            raise ValueError("strict AITER CK core materializes each unpadded logical row")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be 4-D [B,H,S,D]")
        if (
            not allow_logical_group_batch
            and (q.size(0) != 1 or k.size(0) != 1 or v.size(0) != 1)
        ):
            raise ValueError("strict AITER CK core executes one logical batch row at a time")
        if q.size(0) <= 0 or k.size(0) != q.size(0) or v.size(0) != q.size(0):
            raise ValueError("q/k/v must carry the same positive batch size")
        if allow_logical_group_batch and k.size(1) != 1:
            raise ValueError("grouped strict decode requires one KV group per AITER batch row")
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
