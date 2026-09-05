# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strict deterministic GEMM facade for ROCm."""

from __future__ import annotations

import os
from collections.abc import Callable
from threading import Lock

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.triton.matmul.det_gemm import (
    TritonDetGemmOp,
    _triton_gemm_fp32,
    _triton_tree_gemm,
    deterministic_gemm_triton,
)
from rl_engine.runtime_mode import rl_kernel_mode, route_report_enabled

_BACKEND_ENV = "RL_KERNEL_DET_GEMM_BACKEND"
_AUTO_BACKEND = "auto"
_TRITON_BACKEND = "triton"
_ROUTE_REPORTED = False
_ROUTE_REPORT_LOCK = Lock()


def _requested_det_gemm_backend() -> str:
    value = os.getenv(_BACKEND_ENV, _AUTO_BACKEND).strip().lower()
    value = {"rocm": _TRITON_BACKEND}.get(value, value)
    if value not in {_AUTO_BACKEND, _TRITON_BACKEND}:
        raise RuntimeError(
            f"{_BACKEND_ENV} must be '{_AUTO_BACKEND}' or "
            f"'{_TRITON_BACKEND}' on ROCm, got {value!r}"
        )
    return value


_REQUESTED_BACKEND = _requested_det_gemm_backend()


def det_gemm_backend() -> str:
    """Return the strict ROCm GEMM implementation."""

    return _TRITON_BACKEND


def det_gemm_fallback_reason() -> str | None:
    return None


def det_gemm_backend_id() -> str:
    return "rlkernel.det_gemm.triton_tree_rocm.v1"


def _report_strict_route_once() -> None:
    global _ROUTE_REPORTED
    if torch._dynamo.is_compiling() or not route_report_enabled():
        return
    with _ROUTE_REPORT_LOCK:
        if _ROUTE_REPORTED:
            return
        _ROUTE_REPORTED = True
    print(
        f"[RL-Kernel][route] mode={rl_kernel_mode().value} module=gemm "
        f"requested={_REQUESTED_BACKEND} "
        f"actual={det_gemm_backend_id()} fallback=false",
        flush=True,
    )


def det_gemm_linear(
    a: torch.Tensor,
    weight: torch.Tensor,
    *,
    native_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a native [N,K] weight through the strict ROCm backend."""

    del native_op
    return _triton_tree_gemm(a, weight.t().contiguous(), out=out)


def det_gemm_linear_input_gradient(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    *,
    native_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute ``dX = dY @ weight`` through the strict ROCm backend."""

    del native_op
    return deterministic_gemm_triton(grad_output, weight)


def det_gemm_linear_weight_gradient(
    a: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    native_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute ``dWeight = dY.T @ X`` through the strict ROCm backend."""

    del native_op
    return _triton_tree_gemm(
        a.t(),
        grad_output,
        transpose_output=True,
        preserve_a_strides=True,
    )


class _DetLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, weight):
        ctx.save_for_backward(a, weight)
        return det_gemm_linear(a, weight)

    @staticmethod
    def backward(ctx, grad_out):
        a, weight = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if grad_out.dtype != torch.bfloat16:
            grad_out = grad_out.to(torch.bfloat16)
        da = (
            det_gemm_linear_input_gradient(grad_out, weight)
            if ctx.needs_input_grad[0]
            else None
        )
        dweight = (
            det_gemm_linear_weight_gradient(a, grad_out)
            if ctx.needs_input_grad[1]
            else None
        )
        record_backward(
            "det_gemm",
            kernel_id=det_gemm_backend_id(),
            impl="strict_det_gemm",
            family="rocm",
        )
        return da, dweight


class RocmDetGemmOp:
    """Batch-invariant deterministic GEMM backed by the ROCm Triton tree."""

    def __init__(self):
        det_gemm_backend()
        self._triton = TritonDetGemmOp()
        self.op = self._triton
        self.has_hardware_op = True
        _report_strict_route_once()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and b.is_cuda, "Inputs must be on ROCm device"
        return deterministic_gemm_triton(a.contiguous(), b.contiguous())

    def linear(
        self,
        a: torch.Tensor,
        weight: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a native [N,K] linear weight without changing the GEMM tree."""

        assert a.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and weight.is_cuda, "Inputs must be on ROCm device"
        a = a.contiguous()
        weight = weight.contiguous()
        if out is not None:
            if torch.is_grad_enabled() and (a.requires_grad or weight.requires_grad):
                raise RuntimeError("direct-output deterministic GEMM is inference-only")
            return det_gemm_linear(a, weight, out=out)
        return _DetLinearFn.apply(a, weight)

    def forward_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and b.is_cuda, "Inputs must be on ROCm device"
        return _triton_gemm_fp32(a, b)

    def forward_accum_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("FP32-accumulation GEMM requires BF16 or FP32 inputs")
        assert a.is_cuda and b.is_cuda, "Inputs must be on ROCm device"
        return _triton_gemm_fp32(a, b).to(a.dtype)

    def parameter_vjp_contributions_fp32(self, *, a, b, grad_output):
        del b
        rows_a = a.float()
        rows_g = grad_output.float()
        return {"b": rows_a[:, :, None] * rows_g[:, None, :]}


DetGemmOp = RocmDetGemmOp


def deterministic_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional strict ROCm GEMM entry."""

    return deterministic_gemm_triton(a, b)


__all__ = [
    "DetGemmOp",
    "RocmDetGemmOp",
    "det_gemm_backend",
    "det_gemm_backend_id",
    "det_gemm_fallback_reason",
    "det_gemm_linear",
    "det_gemm_linear_input_gradient",
    "det_gemm_linear_weight_gradient",
    "deterministic_gemm",
]
