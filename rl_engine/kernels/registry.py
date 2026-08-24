# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib
import os
from enum import Enum, EnumMeta
from typing import Any, Dict, Optional, Set, Type

import torch

from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger


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

    # GRPO loss (group reward normalization + clipped surrogate + KL)
    TRITON_GRPO_LOSS = "rl_engine.kernels.ops.triton.loss.grpo_loss.TritonGRPOLossOp"
    PYTORCH_GRPO_LOSS = "rl_engine.kernels.ops.pytorch.loss.grpo_loss.NativeGRPOLossOp"

    # Fused linear log-prob (hidden @ W^T -> selected-token logp, no [N, V] logits)
    CUDA_FUSED_LINEAR_LOGP_SM90 = (
        "rl_engine.kernels.ops.cuda.loss.linear_logp.FusedLinearLogpSM90Op"
    )
    CUDA_BATCH_INVARIANT_LINEAR_LOGP_SM90 = (
        "rl_engine.kernels.ops.cuda.loss.batch_invariant_linear_logp."
        "BatchInvariantLinearLogpSM90Op"
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


class KernelRegistry:
    """
    Central dispatcher for high-performance kernels.
    Handles dynamic routing between CUDA, ROCm, and MUSA backends at runtime.
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
                "kv_cache_attention": [OpBackend.PYTORCH_NATIVE_KV_CACHE_ATTN],
                "grpo_loss": [OpBackend.TRITON_GRPO_LOSS, OpBackend.PYTORCH_GRPO_LOSS],
                "linear_logp": [
                    OpBackend.TRITON_LINEAR_LOGP,
                    OpBackend.PYTORCH_LINEAR_LOGP,
                ],
                # Strict single-card contract: no non-invariant fallback.
                "batch_invariant_linear_logp": [
                    OpBackend.CUDA_BATCH_INVARIANT_LINEAR_LOGP_SM90,
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
                "kv_cache_attention": [OpBackend.PYTORCH_NATIVE_KV_CACHE_ATTN],
                "grpo_loss": [OpBackend.TRITON_GRPO_LOSS, OpBackend.PYTORCH_GRPO_LOSS],
                "rope": [OpBackend.TRITON_ROPE, OpBackend.PYTORCH_NATIVE_ROPE],
                "linear_logp": [OpBackend.TRITON_LINEAR_LOGP, OpBackend.PYTORCH_LINEAR_LOGP],
                "batch_invariant_linear_logp": [],
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
            "musa": {
                "logp": [OpBackend.PYTORCH_NATIVE],
                "logp_indexed": [OpBackend.PYTORCH_NATIVE],
                "logp_online": [OpBackend.PYTORCH_NATIVE],
                "logp_online_indexed": [OpBackend.PYTORCH_NATIVE],
                "logp_deterministic": [OpBackend.PYTORCH_NATIVE],
                "logp_deterministic_indexed": [OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.PYTORCH_ATTN],
                "attention": [OpBackend.PYTORCH_NATIVE_ATTENTION],
                "kv_cache_attention": [OpBackend.PYTORCH_NATIVE_KV_CACHE_ATTN],
                "grpo_loss": [OpBackend.PYTORCH_GRPO_LOSS],
                "rope": [OpBackend.PYTORCH_NATIVE_ROPE],
                "linear_logp": [OpBackend.PYTORCH_LINEAR_LOGP],
                "batch_invariant_linear_logp": [],
                "ratio_kl": [OpBackend.PYTORCH_RATIO_KL],
                "pack": [OpBackend.PYTORCH_PACK],
                "det_gemm": [],
                "batch_invariant_logp": [OpBackend.PYTORCH_BATCH_INVARIANT_LOGP],
                "matmul": [OpBackend.PYTORCH_NATIVE_MATMUL],
                "rms_norm": [OpBackend.PYTORCH_NATIVE_RMS_NORM],
                "lm_head": [OpBackend.PYTORCH_NATIVE_LM_HEAD],
                "embedding": [OpBackend.PYTORCH_NATIVE_EMBEDDING],
                "silu": [OpBackend.PYTORCH_NATIVE_SILU],
                "swiglu": [OpBackend.PYTORCH_NATIVE_SWIGLU],
            },
            "cpu": {
                "logp": [OpBackend.PYTORCH_NATIVE],
                "logp_deterministic": [OpBackend.PYTORCH_NATIVE],
                "logp_deterministic_indexed": [OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.PYTORCH_ATTN],
                "attention": [OpBackend.PYTORCH_NATIVE_ATTENTION],
                "kv_cache_attention": [OpBackend.PYTORCH_NATIVE_KV_CACHE_ATTN],
                "grpo_loss": [OpBackend.PYTORCH_GRPO_LOSS],
                "rope": [OpBackend.PYTORCH_NATIVE_ROPE],
                "linear_logp": [OpBackend.PYTORCH_LINEAR_LOGP],
                "batch_invariant_linear_logp": [],
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
        logger.info(f"KernelRegistry initialized for {device_ctx.device_type}")
        self._adjust_priority_for_hardware()
        self._adjust_priority_from_env()

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
            if not self._backend_supports_device(backend, device):
                continue
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

    @staticmethod
    def _backend_supports_device(
        backend: OpBackend,
        device: torch.device | str | None,
    ) -> bool:
        if backend is not OpBackend.CUDA_BATCH_INVARIANT_LINEAR_LOGP_SM90:
            return True
        if torch.version.hip is not None:
            return False

        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        if not _EXT_AVAILABLE or not hasattr(_C, "batch_invariant_linear_logp_sm90"):
            return False

        resolved = torch.device("cuda" if device is None else device)
        if resolved.type != "cuda":
            return False
        try:
            return torch.cuda.get_device_capability(resolved)[0] == 9
        except Exception as error:
            logger.warning(
                "Failed to probe %s for batch-invariant linear_logp: %s",
                resolved,
                error,
            )
            return False

    def _platform_for_device(self, device: torch.device | str | None) -> str:
        if device is None:
            if device_ctx.is_rocm:
                return "rocm"
            if device_ctx.is_musa:
                return "musa"
            if device_ctx.device_type == "cuda":
                return "cuda"
            return "cpu"

        resolved = torch.device(device)
        if resolved.type == "cuda":
            return "rocm" if torch.version.hip is not None else "cuda"
        if resolved.type == "musa":
            return "musa"
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
