# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic BF16 ``SiLU(gate) * up`` forward kernel, Triton backend."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor

from rl_engine.utils.logger import logger

_BLOCK_SIZE = 256


@triton.jit
def _swiglu_forward_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
    output = (gate * sigmoid_gate) * up
    tl.store(output_ptr + offsets, output, mask=mask)


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


class TritonSwiGLUOp(nn.Module):
    """``gate, up -> SiLU(gate) * up`` with BF16 I/O and FP32 element math."""

    op_class = "elementwise"

    def __init__(self) -> None:
        super().__init__()
        logger.info("TritonSwiGLUOp ready (fixed elementwise schedule, no autotune).")

    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        _validate_inputs(gate, up)
        gate = gate.contiguous()
        up = up.contiguous()
        output = torch.empty_like(gate)
        if gate.numel() == 0:
            return output
        grid = (triton.cdiv(gate.numel(), _BLOCK_SIZE),)
        _swiglu_forward_kernel[grid](
            gate,
            up,
            output,
            gate.numel(),
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=4,
        )
        return output
