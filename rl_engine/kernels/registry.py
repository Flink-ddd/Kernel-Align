# SPDX-License-Identifier: Apache-2.0  
# Copyright (c) 2026 RL-Engine Contributors

from enum import Enum, EnumMeta
from typing import Optional, Dict, Any
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger

class _KernelEnumMeta(EnumMeta):
    """Metaclass to provide enhanced error messaging for backend lookups."""
    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            valid_ops = ", ".join(cls.__members__.keys())
            raise ValueError(
                f"Operator '{name}' not found. Supported backends: {valid_ops}"
            )

class OpBackend(Enum, metaclass=_KernelEnumMeta):
    # NVIDIA optimized stack
    FLASH_ATTN = "rl_engine.kernels.cuda.flash_attn.FlashAttentionOp"
    FLASHINFER = "rl_engine.kernels.cuda.flashinfer.FlashInferOp"
    
    # AMD ROCm optimized stack
    ROCM_AITER = "rl_engine.kernels.rocm.aiter.AiterOp"
    ROCM_CK = "rl_engine.kernels.rocm.composable_kernel.CKOp"
    
    # Generic fallback
    TRITON_GENERIC = "rl_engine.kernels.triton.generic.TritonOp"

class KernelRegistry:
    """
    Central dispatcher for high-performance kernels.
    Handles dynamic routing between ROCm and CUDA backends at runtime.
    """
    def __init__(self):
        self._runtime_overrides: Dict[OpBackend, str] = {}
        logger.info(f"KernelRegistry initialized for {device_ctx.device_type}")

    def dispatch(self, op_type: str) -> str:
        """
        Logic to select the optimal backend based on hardware and op_type.
        Critical for resolving VRAM bottlenecks in GRPO-style sampling.
        """
        if device_ctx.is_rocm:
            # Routing logic for AMD hardware
            if "logp" in op_type: 
                return OpBackend.ROCM_AITER.name
            return OpBackend.ROCM_CK.name
        else:
            # Routing logic for NVIDIA hardware
            if "logp" in op_type: 
                return OpBackend.FLASHINFER.name
            return OpBackend.FLASH_ATTN.name

    def get_op_path(self, backend: OpBackend) -> str:
        """Resolve the class path for a given backend, respecting overrides."""
        return self._runtime_overrides.get(backend, backend.value)

kernel_registry = KernelRegistry()