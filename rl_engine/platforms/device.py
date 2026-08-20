# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.platforms.constants import DeviceType
from rl_engine.utils.logger import logger


def _npu_available() -> bool:
    """torch.npu only exists after torch_npu is imported; probe defensively."""
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


class DeviceContext:
    """
    Hardware-aware context manager for high-performance RL tasks.

    Provides transparent support for AMD (ROCm/HIP), NVIDIA (CUDA) and Huawei
    Ascend (NPU/CANN) architectures to ensure backend-agnostic scaling for RL
    operators.
    """

    def __init__(self):
        self.device = torch.device(
            DeviceType.CUDA.value if torch.cuda.is_available() else DeviceType.CPU.value
        )
        self.is_rocm = False
        self.backend_version = "N/A"
        self.device_type = DeviceType.CPU.value

        if self.device.type == DeviceType.CUDA.value:
            # Distinct detection for AMD HIP and  NVIDIA CUDA
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                self.is_rocm = True
                self.device_type = DeviceType.ROCM.value
                self.backend_version = torch.version.hip
                logger.info_once(
                    f"RL-Engine initialized with AMD ROCm backend (Version: {self.backend_version})"
                )
            else:
                self.is_rocm = False
                self.device_type = DeviceType.CUDA.value
                self.backend_version = torch.version.cuda
                logger.info_once(
                    f"RL-Engine initialized with NVIDIA CUDA backend"
                    f" (Version: {self.backend_version})"
                )
        else:
            if _npu_available():
                self.device = torch.device(DeviceType.NPU.value)
                self.device_type = DeviceType.NPU.value
                self.backend_version = getattr(torch.version, "cann", None) or "N/A"
                logger.info_once(
                    f"RL-Engine initialized with Huawei Ascend NPU backend"
                    f" (CANN Version: {self.backend_version})"
                )
            else:
                self.device_type = DeviceType.CPU.value
                logger.warning("No GPU detected. RL-Engine is falling back to CPU mode.")

    def get_preferred_dtype(self):
        """
        Returns the optimal data type for the current hardware.
        AMD ROCm typically yields better performance with bfloat16 in RL workloads.
        """
        return torch.bfloat16 if self.is_rocm else torch.float16


device_ctx = DeviceContext()
