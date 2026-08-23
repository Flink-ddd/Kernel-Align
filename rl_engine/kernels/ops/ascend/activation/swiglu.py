# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Ascend C SwiGLU matching the WS1 FP32-compute activation contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.utils.logger import logger

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False


def _fallback_op():
    from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSwiGLUOp

    return NativeSwiGLUOp()


def _validate_inputs(gate: Tensor, up: Tensor) -> None:
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must share shape, got tuple(gate.shape)={tuple(gate.shape)} "
            f"vs tuple(up.shape)={tuple(up.shape)}"
        )
    if gate.device != up.device:
        raise RuntimeError(
            f"gate and up must be on the same device, got '{gate.device}' and '{up.device}'."
        )
    if gate.dtype not in _SUPPORTED_DTYPES or up.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"gate and up must have dtype fp16, bf16, or fp32, got {gate.dtype} and {up.dtype}."
        )
    if gate.dtype != up.dtype:
        raise TypeError(f"gate and up must share dtype, got {gate.dtype} and {up.dtype}.")


class _SwiGLUAscendFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        output = _C_npu.swiglu_ascend_forward(gate_c, up_c)
        ctx.save_for_backward(gate_c, up_c)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        gate, up = ctx.saved_tensors
        d_gate = d_up = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grads = _C_npu.swiglu_ascend_backward(grad_output.contiguous(), gate, up)
            if ctx.needs_input_grad[0]:
                d_gate = grads[0]
            if ctx.needs_input_grad[1]:
                d_up = grads[1]
        return d_gate, d_up


class SwiGLUAscendOp:
    """Ascend C SwiGLU: ``silu(gate) * up``, with FP32 arithmetic."""

    op_class = "elementwise"

    def __init__(self) -> None:
        required = ("swiglu_ascend_forward", "swiglu_ascend_backward")
        if not _NPU_EXT_AVAILABLE or any(not hasattr(_C_npu, name) for name in required):
            raise RuntimeError(
                "Ascend SwiGLU kernels are not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C_npu SwiGLU kernels.")

    def __call__(self, gate: Tensor, up: Tensor) -> Tensor:
        return self.forward(gate, up)

    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        _validate_inputs(gate, up)
        if gate.device.type != "npu":
            return _fallback_op().forward(gate, up)
        return _SwiGLUAscendFunction.apply(gate, up)

    def forward_fp32(self, gate: Tensor, up: Tensor) -> Tensor:
        _validate_inputs(gate, up)
        if gate.device.type != "npu":
            return _fallback_op().forward_fp32(gate, up)
        return _SwiGLUAscendFunction.apply(gate.float(), up.float())
