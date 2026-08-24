# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib
import os
from enum import Enum, EnumMeta
from typing import Any, Dict, Optional, Set, Type

import torch

from rl_engine.kernels.attention_contract import (
    AttentionBackendCapability,
    AttentionContract,
    AttentionContractError,
    AttentionDispatchResult,
    AttentionDType,
    AttentionMode,
    AttentionRole,
)
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger

# Dispatch policy keywords accepted by ``get_attention_op`` in place of an
# exact backend id. They mirror ``AttentionBackendCapability.implementation_kind``.
ATTENTION_IMPLEMENTATION_KINDS = frozenset({"production", "reference", "deterministic"})


class _KernelEnumMeta(EnumMeta):
    """Metaclass to provide enhanced error messaging for backend lookups."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError as e:
            valid_ops = ", ".join(cls.__members__.keys())
            raise ValueError(f"Operator '{name}' not found. Supported backends: {valid_ops}") from e


class OpBackend(Enum, metaclass=_KernelEnumMeta):
    # NVIDIA optimized stack
    FLASH_ATTN = "rl_engine.kernels.ops.cuda.attention.flash_attn.FlashAttentionOp"
    FLASHINFER = "rl_engine.kernels.ops.cuda.flashinfer.FlashInferOp"

    # TMA-accelerated LogP for SM90+ (Warp Specialization)
    CUDA_FUSED_LOGP_SM90 = "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpSM90Op"
    CUDA_FUSED_LOGP_GENERIC = "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpGenericOp"
    CUDA_DETERMINISTIC_LOGP = "rl_engine.kernels.ops.cuda.loss.logp.DeterministicLogpCUDAOp"
    # Deterministic standard-softmax attention (issue #147); not FlashAttention.
    CUDA_DETERMINISTIC_ATTENTION = (
        "rl_engine.kernels.ops.cuda.attention.deterministic_attn.DeterministicAttentionOp"
    )

    # AMD ROCm optimized stack
    ROCM_AITER = "rl_engine.kernels.ops.rocm.aiter.AiterOp"
    ROCM_CK = "rl_engine.kernels.ops.rocm.composable_kernel.CKOp"
    ROCM_FLASH_ATTN = "rl_engine.kernels.ops.rocm.attention.flash_attn.RocmFlashAttentionOp"
    # WS2 strict ROCm attention core (AITER/CK dense MHA, Split-KV disabled).
    # Registered for contract-aware dispatch only, never in the legacy
    # ``get_op`` priority lists: it is not a drop-in for the SDPA-shaped
    # ``attn`` wrappers and must not be reachable as a silent fallback.
    ROCM_STRICT_ATTENTION = (
        "rl_engine.kernels.ops.rocm.attention.flash_attn.StrictRocmAiterCKAttentionCore"
    )

    # GRPO loss (group reward normalization + clipped surrogate + KL)
    TRITON_GRPO_LOSS = "rl_engine.kernels.ops.triton.loss.grpo_loss.TritonGRPOLossOp"
    PYTORCH_GRPO_LOSS = "rl_engine.kernels.ops.pytorch.loss.grpo_loss.NativeGRPOLossOp"

    # Fused linear log-prob (hidden @ W^T -> selected-token logp, no [N, V] logits)
    CUDA_FUSED_LINEAR_LOGP_SM90 = (
        "rl_engine.kernels.ops.cuda.loss.linear_logp.FusedLinearLogpSM90Op"
    )
    TRITON_LINEAR_LOGP = "rl_engine.kernels.ops.triton.loss.linear_logp.TritonLinearLogpOp"
    PYTORCH_LINEAR_LOGP = "rl_engine.kernels.ops.pytorch.loss.linear_logp.NativeLinearLogpOp"
    # Fused policy-ratio + KL-penalty front-end (PPO/GRPO), logits -> (ratio, kl)
    TRITON_RATIO_KL = "rl_engine.kernels.ops.triton.loss.ratio_kl.TritonRatioKLOp"
    PYTORCH_RATIO_KL = "rl_engine.kernels.ops.pytorch.loss.ratio_kl.NativeRatioKLOp"

    # Variable-length packing (pack-and-pad), [B,S,...] -> [Total_Active,...]
    PYTORCH_PACK = "rl_engine.kernels.ops.pytorch.packing.pack.NativePackOp"
    # Batch-invariant deterministic GEMM (WS1 #146)
    CUDA_DET_GEMM = "rl_engine.kernels.ops.cuda.matmul.det_gemm.DetGemmOp"
    TRITON_DET_GEMM = "rl_engine.kernels.ops.triton.matmul.det_gemm.TritonDetGemmOp"
    # NON-deterministic reference (torch.matmul); reference/benchmark ONLY,
    # intentionally excluded from det_gemm dispatch (cuBLAS breaks invariance).
    PYTORCH_GEMM = "rl_engine.kernels.ops.pytorch.matmul.det_gemm.NativeGemmOp"
    # Batch-invariant selected-logprob (WS1 #148: locked reduction order)
    TRITON_BATCH_INVARIANT_LOGP = (
        "rl_engine.kernels.ops.triton.loss.batch_invariant_logp.TritonBatchInvariantLogpOp"
    )
    PYTORCH_BATCH_INVARIANT_LOGP = (
        "rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp.NativeBatchInvariantLogpOp"
    )
    CUDA_BATCH_INVARIANT_LOGP_SM90 = (
        "rl_engine.kernels.ops.cuda.loss.batch_invariant_logp.BatchInvariantLogpSM90Op"
    )
    ASCEND_BATCH_INVARIANT_LOGP = (
        "rl_engine.kernels.ops.ascend.loss.batch_invariant_logp.BatchInvariantLogpAscendOp"
    )

    # RMSNorm(pre-norm / QK-Norm) - pure Pytorch reference(ws1 ground-truth)
    PYTORCH_NATIVE_RMS_NORM = "rl_engine.kernels.ops.pytorch.norm.rms_norm.NativeRMSNormOp"

    # Generic fallback
    TRITON_GENERIC = "rl_engine.kernels.ops.triton.generic.TritonOp"
    PYTORCH_ATTN = "rl_engine.kernels.ops.pytorch.attention.NativeAttentionOp"
    PYTORCH_NATIVE = "rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp"
    PYTORCH_NATIVE_MATMUL = "rl_engine.kernels.ops.pytorch.linear.matmul.NativeMatmulOp"
    PYTORCH_NATIVE_ROPE = "rl_engine.kernels.ops.pytorch.rotary_embedding.rope.NativeRoPEOp"
    TRITON_ROPE = "rl_engine.kernels.ops.triton.rotary_embedding.rope.TritonRoPEOp"
    CUDA_ROPE_SM90 = "rl_engine.kernels.ops.cuda.rotary_embedding.rope.RoPESM90Op"
    PYTORCH_NATIVE_SILU = "rl_engine.kernels.ops.pytorch.activation.swiglu.NativeSiLUOp"
    PYTORCH_NATIVE_SWIGLU = "rl_engine.kernels.ops.pytorch.activation.swiglu.NativeSwiGLUOp"
    CUDA_SILU = "rl_engine.kernels.ops.cuda.activation.swiglu.SiLUCudaOp"
    CUDA_SWIGLU = "rl_engine.kernels.ops.cuda.activation.swiglu.SwiGLUCudaOp"
    TRITON_SILU = "rl_engine.kernels.ops.triton.activation.swiglu.TritonSiLUOp"
    TRITON_SWIGLU = "rl_engine.kernels.ops.triton.activation.swiglu.TritonSwiGLUOp"

    # WS1 pure-PyTorch ground-truth attention reference (hand-written fp32 softmax).
    # Distinct from PYTORCH_ATTN above, which is the production SDPA fallback.
    PYTORCH_NATIVE_ATTENTION = (
        "rl_engine.kernels.ops.pytorch.attention.standard_attn.NativeAttentionOp"
    )
    # WS2 correctness-first CP attention reference. Keep this separate from
    # ``attention`` so production dispatch never selects the reference by accident.
    PYTORCH_CP_ATTENTION = (
        "rl_engine.kernels.ops.pytorch.attention.cp_attention."
        "DeterministicCPAttentionReferenceOp"
    )
    # WS1 pure-PyTorch ground-truth KV-cache (decode/incremental) attention
    # reference; concats cache+new then reuses the standard attention reduction.
    PYTORCH_NATIVE_KV_CACHE_ATTN = (
        "rl_engine.kernels.ops.pytorch.attention.kv_cache.NativeKVCacheAttnOp"
    )
    # WS1 pure-PyTorch ground-truth linear ops
    PYTORCH_NATIVE_LM_HEAD = "rl_engine.kernels.ops.pytorch.linear.lm_head.NativeLMHeadOp"
    # WS1 pure-PyTorch ground-truth embedding ops
    PYTORCH_NATIVE_EMBEDDING = "rl_engine.kernels.ops.pytorch.linear.embedding.NativeEmbeddingOp"
    CUDA_SM90_LM_HEAD = "rl_engine.kernels.ops.cuda.linear.lm_head.SM90LMHeadOp"
    CUDA_SM90_EMBEDDING = "rl_engine.kernels.ops.cuda.linear.embedding.SM90EmbeddingOp"


def resolve_logp_op_type(
    logp_backend: Optional[str] = None,
    *,
    require_batch_invariant: bool = False,
) -> str:
    """Normalize user-facing logp backend names to KernelRegistry op types."""

    normalized = (logp_backend or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "logp",
        "default": "logp",
        "fused": "logp",
        "fused_logp": "logp",
        "generic": "logp",
        "logp": "logp",
        "indexed": "logp_indexed",
        "logp_indexed": "logp_indexed",
        "online": "logp_online",
        "logp_online": "logp_online",
        "online_indexed": "logp_online_indexed",
        "logp_online_indexed": "logp_online_indexed",
        "deterministic": "logp_deterministic",
        "deterministic_cuda": "logp_deterministic",
        "batch_invariant": "logp_deterministic",
        "batch_invariant_deterministic": "logp_deterministic",
        "logp_deterministic": "logp_deterministic",
        "deterministic_indexed": "logp_deterministic_indexed",
        "deterministic_cuda_indexed": "logp_deterministic_indexed",
        "batch_invariant_indexed": "logp_deterministic_indexed",
        "logp_deterministic_indexed": "logp_deterministic_indexed",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"unsupported logp backend {logp_backend!r}; valid values: {valid}")

    op_type = aliases[normalized]
    if require_batch_invariant:
        if normalized in {"auto", "default", "logp"}:
            return "logp_deterministic"
        if not op_type.startswith("logp_deterministic"):
            raise ValueError(
                "require_batch_invariant_logp=True requires a deterministic logp backend; "
                f"got {logp_backend!r}"
            )
    return op_type


def _rocm_strict_attention_available() -> bool:
    """Return whether the strict ROCm AITER/CK attention core can actually load.

    The probe mirrors the ROCm logprob backend gate: it must be true only when
    this process can really execute the strict arithmetic, so an unavailable
    vendor stack results in the backend never being registered rather than in a
    registered backend that fails at materialization time.
    """

    if torch.version.hip is None:
        return False
    try:
        from rl_engine.kernels.ops.rocm.attention.flash_attn import _load_aiter_ck_ops
    except ImportError:
        return False
    try:
        _load_aiter_ck_ops()
    except Exception:  # StrictRocmAttentionUnavailable and vendor import errors
        return False
    return True


class KernelRegistry:
    """
    Central dispatcher for high-performance kernels.
    Handles dynamic routing between ROCm and CUDA backends at runtime.
    """

    def __init__(self):
        self._instance_cache: Dict[str, Any] = {}
        self._failed_backends: Set[str] = set()

        self._priority_map = {
            "cuda": {
                "logp": [
                    OpBackend.CUDA_FUSED_LOGP_GENERIC,
                    OpBackend.FLASHINFER,
                    OpBackend.TRITON_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_indexed": [
                    OpBackend.CUDA_FUSED_LOGP_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_online": [
                    OpBackend.CUDA_FUSED_LOGP_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_online_indexed": [
                    OpBackend.CUDA_FUSED_LOGP_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_deterministic": [
                    OpBackend.CUDA_DETERMINISTIC_LOGP,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_deterministic_indexed": [
                    OpBackend.CUDA_DETERMINISTIC_LOGP,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "attn": [OpBackend.FLASH_ATTN, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_ATTN],
                "attention": [
                    OpBackend.CUDA_DETERMINISTIC_ATTENTION,
                    OpBackend.PYTORCH_NATIVE_ATTENTION,
                ],
                "cp_attention": [OpBackend.PYTORCH_CP_ATTENTION],
                "kv_cache_attention": [OpBackend.PYTORCH_NATIVE_KV_CACHE_ATTN],
                "grpo_loss": [OpBackend.TRITON_GRPO_LOSS, OpBackend.PYTORCH_GRPO_LOSS],
                "linear_logp": [
                    OpBackend.TRITON_LINEAR_LOGP,
                    OpBackend.PYTORCH_LINEAR_LOGP,
                ],
                "ratio_kl": [OpBackend.TRITON_RATIO_KL, OpBackend.PYTORCH_RATIO_KL],
                "pack": [OpBackend.PYTORCH_PACK],
                "det_gemm": [OpBackend.CUDA_DET_GEMM, OpBackend.TRITON_DET_GEMM],
                "batch_invariant_logp": [
                    OpBackend.TRITON_BATCH_INVARIANT_LOGP,
                    OpBackend.PYTORCH_BATCH_INVARIANT_LOGP,
                ],
                "rms_norm": [OpBackend.PYTORCH_NATIVE_RMS_NORM],
                "lm_head": [OpBackend.PYTORCH_NATIVE_LM_HEAD],
                "embedding": [OpBackend.PYTORCH_NATIVE_EMBEDDING],
                "silu": [
                    OpBackend.CUDA_SILU,
                    OpBackend.TRITON_SILU,
                    OpBackend.PYTORCH_NATIVE_SILU,
                ],
                "swiglu": [
                    OpBackend.CUDA_SWIGLU,
                    OpBackend.TRITON_SWIGLU,
                    OpBackend.PYTORCH_NATIVE_SWIGLU,
                ],
                # Default dispatch logic for new operators
                "matmul": [OpBackend.PYTORCH_NATIVE_MATMUL],
                "rope": [
                    OpBackend.CUDA_ROPE_SM90,
                    OpBackend.TRITON_ROPE,
                    OpBackend.PYTORCH_NATIVE_ROPE,
                ],
            },
            "rocm": {
                "logp": [
                    OpBackend.ROCM_AITER,
                    OpBackend.TRITON_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_deterministic": [OpBackend.PYTORCH_NATIVE],
                "logp_deterministic_indexed": [OpBackend.PYTORCH_NATIVE],
                "attn": [
                    OpBackend.ROCM_FLASH_ATTN,
                    OpBackend.PYTORCH_ATTN,
                    OpBackend.TRITON_GENERIC,
                ],
                "attention": [OpBackend.PYTORCH_NATIVE_ATTENTION],
                "cp_attention": [OpBackend.PYTORCH_CP_ATTENTION],
                "kv_cache_attention": [OpBackend.PYTORCH_NATIVE_KV_CACHE_ATTN],
                "grpo_loss": [OpBackend.TRITON_GRPO_LOSS, OpBackend.PYTORCH_GRPO_LOSS],
                "rope": [OpBackend.TRITON_ROPE, OpBackend.PYTORCH_NATIVE_ROPE],
                "linear_logp": [OpBackend.TRITON_LINEAR_LOGP, OpBackend.PYTORCH_LINEAR_LOGP],
                "ratio_kl": [OpBackend.TRITON_RATIO_KL, OpBackend.PYTORCH_RATIO_KL],
                "pack": [OpBackend.PYTORCH_PACK],
                "det_gemm": [OpBackend.TRITON_DET_GEMM],
                "batch_invariant_logp": [
                    OpBackend.TRITON_BATCH_INVARIANT_LOGP,
                    OpBackend.PYTORCH_BATCH_INVARIANT_LOGP,
                ],
                "matmul": [OpBackend.PYTORCH_NATIVE_MATMUL],
                "rms_norm": [OpBackend.PYTORCH_NATIVE_RMS_NORM],
                "lm_head": [OpBackend.PYTORCH_NATIVE_LM_HEAD],
                "embedding": [OpBackend.PYTORCH_NATIVE_EMBEDDING],
                "silu": [OpBackend.TRITON_SILU, OpBackend.PYTORCH_NATIVE_SILU],
                "swiglu": [OpBackend.TRITON_SWIGLU, OpBackend.PYTORCH_NATIVE_SWIGLU],
            },
            "cpu": {
                "logp": [OpBackend.PYTORCH_NATIVE],
                "logp_deterministic": [OpBackend.PYTORCH_NATIVE],
                "logp_deterministic_indexed": [OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.PYTORCH_ATTN],
                "attention": [OpBackend.PYTORCH_NATIVE_ATTENTION],
                "cp_attention": [OpBackend.PYTORCH_CP_ATTENTION],
                "kv_cache_attention": [OpBackend.PYTORCH_NATIVE_KV_CACHE_ATTN],
                "grpo_loss": [OpBackend.PYTORCH_GRPO_LOSS],
                "rope": [OpBackend.PYTORCH_NATIVE_ROPE],
                "linear_logp": [OpBackend.PYTORCH_LINEAR_LOGP],
                "ratio_kl": [OpBackend.PYTORCH_RATIO_KL],
                "pack": [OpBackend.PYTORCH_PACK],
                "batch_invariant_logp": [OpBackend.PYTORCH_BATCH_INVARIANT_LOGP],
                "matmul": [OpBackend.PYTORCH_NATIVE_MATMUL],
                "rms_norm": [OpBackend.PYTORCH_NATIVE_RMS_NORM],
                "lm_head": [OpBackend.PYTORCH_NATIVE_LM_HEAD],
                "embedding": [OpBackend.PYTORCH_NATIVE_EMBEDDING],
                "silu": [OpBackend.PYTORCH_NATIVE_SILU],
                "swiglu": [OpBackend.PYTORCH_NATIVE_SWIGLU],
            },
        }
        # Preserve the former CPU fallback behavior for every operator on NPU,
        # then override only the operator with an Ascend-specific backend.
        self._priority_map["npu"] = {
            op_type: candidates.copy() for op_type, candidates in self._priority_map["cpu"].items()
        }
        self._priority_map["npu"]["batch_invariant_logp"] = [
            OpBackend.ASCEND_BATCH_INVARIANT_LOGP,
            OpBackend.PYTORCH_BATCH_INVARIANT_LOGP,
        ]
        logger.info(f"KernelRegistry initialized for {device_ctx.device_type}")
        self._adjust_priority_for_hardware()
        self._adjust_priority_from_env()

        # WS2 attention dispatch owns its own candidate list and starts empty on
        # every platform. The legacy ``attn`` / ``attention`` priority lists stay
        # untouched, so a strict WS2 caller can never be served by an SDPA-shaped
        # wrapper that does not export attention-domain LSE, and the strict core
        # can never be picked up by a legacy caller as a silent substitution.
        self._attention_candidates: Dict[str, list] = {
            platform: [] for platform in self._priority_map
        }
        self._attention_capabilities: Dict[str, Dict[OpBackend, AttentionBackendCapability]] = {
            platform: {} for platform in self._priority_map
        }

        # Strict ROCm production core: AITER/CK dense MHA with Split-KV disabled.
        # Registered only when the vendor entry points genuinely load, so an
        # explicit request on a machine without AITER fails loudly in
        # ``get_attention_op`` instead of resolving to different arithmetic.
        rocm_strict_attention_capability = AttentionBackendCapability(
            backend_id="aiter.rocm.ck_dense_mha",
            roles=frozenset({AttentionRole.TRAIN, AttentionRole.INFER}),
            modes=frozenset({AttentionMode.PREFILL, AttentionMode.CHUNKED_PREFILL}),
            dtypes=frozenset({AttentionDType.BF16, AttentionDType.FP16}),
            cp_world_sizes=(1,),
            tp_world_sizes=None,
            exports_attention_lse=True,
            deterministic_cp_merge=False,
            supports_packed_varlen=False,
            supports_kv_cache=False,
            supports_rope_metadata=True,
            supports_fused_rope_attention=False,
            supports_split_kv_disabled=True,
            supports_split_kv_fixed=False,
            supports_split_kv_auto=False,
            reports_actual_split_kv_plan=True,
            implementation_kind="production",
        )
        if _rocm_strict_attention_available() and "rocm" in self._priority_map:
            self.register_attention_backend(
                OpBackend.ROCM_STRICT_ATTENTION,
                rocm_strict_attention_capability,
                platform="rocm",
                prepend=True,
            )

    def register_attention_backend(
        self,
        backend: OpBackend,
        capability: AttentionBackendCapability,
        *,
        platform: Optional[str] = None,
        prepend: bool = False,
    ) -> None:
        """Register (or replace) a backend for WS2 contract-aware attention dispatch.

        This is the supported seam for making an attention backend selectable by
        ``get_attention_op`` without touching the legacy ``get_op`` priority
        lists. Registering the same backend again replaces its capability
        without duplicating the candidate entry.
        """

        if not isinstance(backend, OpBackend):
            raise AttentionContractError("backend must be an OpBackend")
        if not isinstance(capability, AttentionBackendCapability):
            raise AttentionContractError("capability must be an AttentionBackendCapability")
        resolved_platform = platform if platform is not None else self._platform()
        if resolved_platform not in self._priority_map:
            raise AttentionContractError(
                f"unsupported platform {resolved_platform!r}; expected one of "
                f"{sorted(self._priority_map)}"
            )
        candidates = self._attention_candidates.setdefault(resolved_platform, [])
        self._attention_capabilities.setdefault(resolved_platform, {})[backend] = capability
        if backend not in candidates:
            if prepend:
                candidates.insert(0, backend)
            else:
                candidates.append(backend)

    def get_attention_op(
        self,
        contract: AttentionContract,
        *,
        requested_backend: str = "auto",
    ) -> AttentionDispatchResult:
        """Resolve only a backend that explicitly supports the WS2 attention contract.

        Deliberately separate from the legacy ``get_op``: existing callers keep
        their current behavior, while WS2 callers cannot silently fall back to a
        backend with different reduction or Split-KV semantics.

        ``requested_backend`` is either a case-insensitive policy keyword
        (``auto`` | ``production`` | ``reference``) or an exact, case-sensitive
        stable backend id. Strictness comes from the contract's capability
        checks rather than from this policy string. With ``cp_world_size > 1``,
        ``auto`` is rejected: per-rank auto resolution can diverge across ranks,
        so distributed callers must name a policy or backend id.
        """

        if not isinstance(contract, AttentionContract):
            raise AttentionContractError("contract must be an AttentionContract")
        if not isinstance(requested_backend, str) or not requested_backend.strip():
            raise AttentionContractError("requested_backend must be a non-empty string")
        requested_backend = requested_backend.strip()
        if requested_backend.lower() == "deterministic":
            raise AttentionContractError(
                'requested_backend="deterministic" is not a dispatch policy; request '
                "determinism through ReductionSpec/SplitKVSpec and match it against the "
                "backend capability flags instead"
            )
        if requested_backend.lower() == "auto" and contract.sharding.cp_world_size > 1:
            raise AttentionContractError(
                "Unsafe dispatch: requested_backend='auto' is not permitted when "
                "cp_world_size > 1 without explicit cross-rank preflighting."
            )

        platform = self._platform()
        candidates = self._attention_candidates.get(platform, [])
        rejected: list[str] = []
        # provenance["fallback"] reports only capability/load rejections of
        # otherwise-eligible candidates; skips caused purely by the caller's own
        # requested_backend policy filter are not fallbacks.
        capability_rejections = 0

        platform_capabilities = self._attention_capabilities.get(platform, {})
        for backend in candidates:
            capability = platform_capabilities.get(backend)
            if capability is None:
                rejected.append(f"{backend.name}: no AttentionBackendCapability declared")
                capability_rejections += 1
                continue
            policy_mismatch = self._attention_policy_mismatch(requested_backend, capability)
            if policy_mismatch is not None:
                rejected.append(f"{backend.name}: {policy_mismatch}")
                continue
            capability_incompat = list(capability.incompatibilities(contract))
            if capability_incompat:
                rejected.append(f"{backend.name}: " + "; ".join(capability_incompat))
                capability_rejections += 1
                continue

            op = self._get_or_create_backend(backend)
            if op is None:
                rejected.append(f"{backend.name}: backend could not be loaded or instantiated")
                capability_rejections += 1
                continue

            provenance = {
                "requested_backend": requested_backend,
                "actual_backend": capability.backend_id,
                "backend_enum": backend.name,
                "platform": platform,
                "fallback": capability_rejections > 0,
                "fallback_reason": None,
                "prior_rejections": list(rejected),
                "contract": contract.to_dict(),
                "capability": capability.to_dict(),
            }
            return AttentionDispatchResult(op=op, capability=capability, provenance=provenance)

        details = " | ".join(rejected) if rejected else "no candidates registered"
        requested = contract.to_dict()
        raise RuntimeError(
            "No attention backend supports the requested WS2 contract on "
            f"{platform}: role={requested['role']}, mode={requested['mode']}, "
            f"dtype={requested['dtype']}, TP={contract.sharding.tp_world_size}, "
            f"CP={contract.sharding.cp_world_size}, "
            f"split_kv={contract.split_kv.mode.value}. Rejections: {details}"
        )

    @staticmethod
    def _attention_policy_mismatch(
        requested_backend: str,
        capability: AttentionBackendCapability,
    ) -> str | None:
        policy = requested_backend.lower()
        if policy == "auto":
            return None
        if policy in ATTENTION_IMPLEMENTATION_KINDS:
            if capability.implementation_kind == policy:
                return None
            return (
                f"implementation_kind={capability.implementation_kind} does not satisfy "
                f"requested_backend={policy}"
            )
        if capability.backend_id == requested_backend:
            return None
        return (
            f"backend_id={capability.backend_id} does not match "
            f"requested_backend={requested_backend}"
        )

    def _platform(self) -> str:
        return self._platform_for_device(None)

    def _get_or_create_backend(self, backend: OpBackend) -> Any | None:
        if backend.name in self._instance_cache:
            return self._instance_cache[backend.name]
        if backend.name in self._failed_backends:
            return None

        op_class = self._load_backend(backend)
        if op_class is None:
            self._failed_backends.add(backend.name)
            return None
        try:
            op = op_class()
        except Exception as exc:
            logger.error(f"Failed to instantiate {backend.name}: {exc}")
            self._failed_backends.add(backend.name)
            return None
        self._instance_cache[backend.name] = op
        return op

    def _adjust_priority_from_env(self):
        rocm_attn_backend = os.getenv("RL_KERNEL_ROCM_ATTN_BACKEND", "").strip().lower()
        if rocm_attn_backend in {"flash_attn", "flash-attn", "flash_attention"}:
            self._priority_map["rocm"]["attn"] = [
                OpBackend.ROCM_FLASH_ATTN,
                OpBackend.PYTORCH_ATTN,
                OpBackend.TRITON_GENERIC,
            ]
        elif rocm_attn_backend in {"native", "pytorch", "sdpa"}:
            self._priority_map["rocm"]["attn"] = [
                OpBackend.PYTORCH_ATTN,
                OpBackend.ROCM_FLASH_ATTN,
                OpBackend.TRITON_GENERIC,
            ]
        elif rocm_attn_backend and rocm_attn_backend not in {
            "native",
            "pytorch",
            "sdpa",
        }:
            logger.warning(
                "Unknown RL_KERNEL_ROCM_ATTN_BACKEND=%s; using default ROCm attention priority.",
                rocm_attn_backend,
            )

    def _adjust_priority_for_hardware(self):
        """Adjust CUDA priorities for hardware-gated experimental and production kernels."""
        if device_ctx.device_type != "cuda":
            return
        try:
            import torch

            from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

            cc_major, cc_minor = torch.cuda.get_device_capability()
            cc = cc_major * 10 + cc_minor
            tma_compiled = _EXT_AVAILABLE and hasattr(_C, "fused_logp_sm90")

            sm90_logp_enabled = os.getenv("RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP") == "1"
            if sm90_logp_enabled and tma_compiled and cc_major in (9, 10, 12):
                logger.info(
                    f"Detected TMA-capable architecture (SM{cc}); "
                    "prioritizing experimental fused TMA LogP kernel."
                )
                logp_list = self._priority_map["cuda"]["logp"]
                if OpBackend.CUDA_FUSED_LOGP_SM90 not in logp_list:
                    logp_list.insert(0, OpBackend.CUDA_FUSED_LOGP_SM90)

            # The fused linear-logp SM90 kernel uses TMA bulk-tensor copies built
            # for sm_90a -- gate strictly on cc_major == 9 (Hopper), not >= 9.
            linear_logp_compiled = _EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90")
            if linear_logp_compiled and cc_major == 9:
                ll_list = self._priority_map["cuda"]["linear_logp"]
                if OpBackend.CUDA_FUSED_LINEAR_LOGP_SM90 not in ll_list:
                    ll_list.insert(0, OpBackend.CUDA_FUSED_LINEAR_LOGP_SM90)

            # Batch-invariant logp SM90 kernel: same sm_90a TMA gating (Hopper only).
            batch_inv_compiled = _EXT_AVAILABLE and hasattr(_C, "batch_invariant_logp_sm90")
            if batch_inv_compiled and cc_major == 9:
                bi_list = self._priority_map["cuda"]["batch_invariant_logp"]
                if OpBackend.CUDA_BATCH_INVARIANT_LOGP_SM90 not in bi_list:
                    bi_list.insert(0, OpBackend.CUDA_BATCH_INVARIANT_LOGP_SM90)
            elif cc >= 90:
                logger.debug(
                    f"SM{cc}: fused linear-logp SM90 kernel not compiled into _C; "
                    "using generic linear-logp backend."
                )

            sm90_embedding_compiled = _EXT_AVAILABLE and hasattr(_C, "embedding_sm90_forward")
            if sm90_embedding_compiled and cc_major == 9:
                embedding_list = self._priority_map["cuda"]["embedding"]
                if OpBackend.CUDA_SM90_EMBEDDING not in embedding_list:
                    embedding_list.insert(0, OpBackend.CUDA_SM90_EMBEDDING)

            sm90_lm_head_compiled = _EXT_AVAILABLE and hasattr(_C, "lm_head_sm90_forward")
            if sm90_lm_head_compiled and cc_major == 9:
                lm_head_list = self._priority_map["cuda"]["lm_head"]
                if OpBackend.CUDA_SM90_LM_HEAD not in lm_head_list:
                    lm_head_list.insert(0, OpBackend.CUDA_SM90_LM_HEAD)
        except Exception as e:
            logger.warning(f"Failed to probe device capability: {e}")

    def get_op(self, op_type: str, device: torch.device | str | None = None) -> Any:
        """Core distribution logic: Automatically select the best operator
        based on hardware and priority.
        """
        platform = self._platform_for_device(device)
        candidates = self._priority_map.get(platform, {}).get(op_type, [OpBackend.PYTORCH_NATIVE])

        for backend in candidates:
            if backend.name in self._instance_cache:
                return self._instance_cache[backend.name]

            if backend.name in self._failed_backends:
                continue

            op_class = self._load_backend(backend)
            if op_class:
                try:
                    op_instance = op_class()
                    self._instance_cache[backend.name] = op_instance
                    return op_instance
                except Exception as e:
                    logger.error(f"Failed to instantiate {backend.name}: {e}")
                    self._failed_backends.add(backend.name)
            else:
                self._failed_backends.add(backend.name)

        raise RuntimeError(f"No functional backend found for {op_type} on {platform}")

    def _platform_for_device(self, device: torch.device | str | None) -> str:
        if device is None:
            if device_ctx.is_rocm:
                return "rocm"
            if device_ctx.device_type == "cuda":
                return "cuda"
            if device_ctx.device_type == "npu":
                return "npu"
            return "cpu"

        resolved = torch.device(device)
        if resolved.type == "cuda":
            return "rocm" if torch.version.hip is not None else "cuda"
        if resolved.type in self._priority_map:
            return resolved.type
        return "cpu"

    def _load_backend(self, backend: OpBackend) -> Optional[Type]:
        """Dynamic loading technique: Import modules only when needed
        and check environment dependencies.
        """
        module_path, class_name = backend.value.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            missing_module = str(e.name) if hasattr(e, "name") else ""
            is_missing_backend = missing_module and (
                missing_module == module_path or module_path.startswith(missing_module)
            )
            if missing_module and "rl_engine" in missing_module and not is_missing_backend:
                logger.critical(f"Internal wrapper implementation bug in '{module_path}': {e}")
                raise e
            logger.warning(f"Backend {backend.name} unavailable: {e}. Falling back...")
            return None


kernel_registry = KernelRegistry()
