# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic BF16 ``SiLU(gate) * up`` forward kernel for Hopper."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


def _validate_inputs(gate: Tensor, up: Tensor) -> None:
    if gate.device.type != "cuda" or up.device.type != "cuda":
        raise RuntimeError("gate and up must be CUDA tensors")
    if gate.dtype is not torch.bfloat16 or up.dtype is not torch.bfloat16:
        raise TypeError("gate and up must have dtype torch.bfloat16")
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must share shape, got {tuple(gate.shape)} vs {tuple(up.shape)}"
        )
    if gate.device != up.device:
        raise ValueError(f"gate and up must share device, got {gate.device} vs {up.device}")


def _as_aligned_contiguous(tensor: Tensor) -> Tensor:
    tensor = tensor.contiguous()
    # A contiguous view may still begin at an odd BF16 storage offset.
    return tensor if tensor.data_ptr() % 4 == 0 else tensor.clone()


class SwiGLUSM90Op(nn.Module):
    """``gate, up -> SiLU(gate) * up`` with BF16 I/O and FP32 element math."""

    op_class = "elementwise"

    def __init__(self) -> None:
        super().__init__()
        if not _EXT_AVAILABLE or not hasattr(_C, "swiglu_forward_sm90"):
            raise RuntimeError(
                "SM90 SwiGLU is not compiled into rl_engine._C. Rebuild with "
                "'KERNEL_ALIGN_ACTIVATION_SM90=1 pip install --no-build-isolation -e .'."
            )
        logger.info("Successfully linked to the SM90 SwiGLU forward kernel.")

    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        _validate_inputs(gate, up)
        return _C.swiglu_forward_sm90(_as_aligned_contiguous(gate), _as_aligned_contiguous(up))
