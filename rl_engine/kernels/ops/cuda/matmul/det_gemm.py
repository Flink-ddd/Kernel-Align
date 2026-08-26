# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant deterministic GEMM backends (WS1 #146).

Hand-written kernel (csrc/cuda/gemm/det_gemm_kernel.cu): fixed K-accumulation
order, FP32 accumulation, no split-K. On Hopper, an explicitly configured
cuBLASLt no-split-K backend is also available. Both are strict implementations;
backend selection never falls back silently. Tensor-parallel GEMM is WS2.
"""
import os
from collections.abc import Callable
from threading import Lock

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.runtime_mode import rl_kernel_mode, route_report_enabled

_BACKEND_ENV = "RL_KERNEL_DET_GEMM_BACKEND"
_SM90_BACKEND = "sm90"
_CUBLASLT_BACKEND = "cublaslt_nosplitk"
_CUBLASLT_CONFIGURED = False
_ROUTE_REPORTED = False
_ROUTE_REPORT_LOCK = Lock()


def det_gemm_backend() -> str:
    """Return the selected strict GEMM implementation.

    The aligned runtime contract selects the same Hopper cuBLASLt path as vLLM.
    Other environments retain the self-contained SM90 backend unless overridden.
    """

    aligned_default = (
        os.getenv("VLLM_BATCH_INVARIANT", "").strip().lower()
        in {"1", "true", "yes", "on"}
        and os.getenv("CUBLAS_WORKSPACE_CONFIG", "").strip() == ":16:8"
        and os.getenv("CUBLASLT_WORKSPACE_SIZE", "").strip() == "1"
    )
    default = _CUBLASLT_BACKEND if aligned_default else _SM90_BACKEND
    value = os.getenv(_BACKEND_ENV, default).strip().lower()
    aliases = {
        "cuda": _SM90_BACKEND,
        "cublaslt": _CUBLASLT_BACKEND,
    }
    value = aliases.get(value, value)
    if value not in {_SM90_BACKEND, _CUBLASLT_BACKEND}:
        raise RuntimeError(
            f"{_BACKEND_ENV} must be '{_SM90_BACKEND}' or "
            f"'{_CUBLASLT_BACKEND}', got {value!r}"
        )
    return value


def det_gemm_backend_id() -> str:
    backend = det_gemm_backend()
    if backend == _CUBLASLT_BACKEND:
        return "rlkernel.det_gemm.cublaslt_nosplitk.v1"
    return "rlkernel.det_gemm.sm90.v1"


def _report_strict_route_once() -> None:
    global _ROUTE_REPORTED
    if not route_report_enabled():
        return
    with _ROUTE_REPORT_LOCK:
        if _ROUTE_REPORTED:
            return
        _ROUTE_REPORTED = True
    print(
        f"[RL-Kernel][route] mode={rl_kernel_mode().value} module=gemm "
        "requested=strict "
        f"actual={det_gemm_backend_id()} fallback=false",
        flush=True,
    )


def _configure_cublaslt_nosplitk(a: torch.Tensor | None = None) -> None:
    """Validate the Hopper batch-invariant cuBLASLt contract once per process."""

    global _CUBLASLT_CONFIGURED
    if _CUBLASLT_CONFIGURED:
        return
    if torch._dynamo.is_compiling():
        raise RuntimeError(
            "cublaslt_nosplitk must be configured before torch.compile tracing"
        )
    device = None if a is None else a.device
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise RuntimeError("cublaslt_nosplitk is currently validated only on SM90")
    workspace = os.getenv("CUBLAS_WORKSPACE_CONFIG", "").strip()
    if workspace != ":16:8":
        raise RuntimeError(
            "cublaslt_nosplitk requires CUBLAS_WORKSPACE_CONFIG=:16:8 before "
            "the process starts"
        )
    if os.getenv("CUBLASLT_WORKSPACE_SIZE", "").strip() != "1":
        raise RuntimeError(
            "cublaslt_nosplitk requires CUBLASLT_WORKSPACE_SIZE=1 before the "
            "process starts"
        )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.preferred_blas_library("cublaslt")
    _CUBLASLT_CONFIGURED = True


def det_gemm_linear(
    a: torch.Tensor,
    weight: torch.Tensor,
    *,
    native_op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply a native [N,K] weight through the selected strict backend."""

    if det_gemm_backend() == _CUBLASLT_BACKEND:
        _configure_cublaslt_nosplitk(a)
        _report_strict_route_once()
        return torch.mm(a, weight.t())
    _report_strict_route_once()
    if native_op is not None:
        return native_op(a, weight)
    return _det_gemm_fwd_weight(a, weight)


@torch.library.custom_op("rl_kernel::det_gemm_fwd_weight", mutates_args=())
def _det_gemm_fwd_weight(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Expose the native-weight SM90 GEMM through the PyTorch dispatcher."""

    return _C.det_gemm_fwd_weight(a, weight)


@_det_gemm_fwd_weight.register_fake
def _det_gemm_fwd_weight_fake(
    a: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return a.new_empty((*a.shape[:-1], weight.shape[0]))


class _DetGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, output_fp32=False):
        ctx.save_for_backward(a, b)
        if output_fp32:
            if not hasattr(_C, "det_gemm_fwd_fp32"):
                raise RuntimeError(
                    "FP32 deterministic GEMM output requires the rebuilt extension"
                )
            return _C.det_gemm_fwd_fp32(a, b)
        return _C.det_gemm_fwd(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        if grad_out.dtype != torch.bfloat16:
            grad_out = grad_out.to(torch.bfloat16)
        da = _C.det_gemm_da(grad_out, b) if ctx.needs_input_grad[0] else None
        db = _C.det_gemm_db(a, grad_out) if ctx.needs_input_grad[1] else None
        record_backward(
            "det_gemm",
            kernel_id="rl_engine._C.det_gemm_da+rl_engine._C.det_gemm_db",
            impl="cuda_det_gemm",
            family="cuda",
        )
        return da, db, None


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
            _C.det_gemm_da_weight(grad_out, weight) if ctx.needs_input_grad[0] else None
        )
        dweight = _C.det_gemm_dw(a, grad_out) if ctx.needs_input_grad[1] else None
        record_backward(
            "det_gemm",
            kernel_id="rl_engine._C.det_gemm_da_weight+rl_engine._C.det_gemm_dw",
            impl="cuda_det_gemm",
            family="cuda",
        )
        return da, dweight


class _DetGemmAccumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if not hasattr(_C, "det_gemm_rowwise_fwd_fp32"):
            raise RuntimeError(
                "FP32 rowwise deterministic GEMM requires the rebuilt SM90 extension"
            )
        return _C.det_gemm_rowwise_fwd_fp32(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_fp32 = grad_out.contiguous().float()
        a_fp32 = a.contiguous().float()
        b_fp32 = b.contiguous().float()
        da = (
            _C.det_gemm_rowwise_fwd_fp32(grad_fp32, b_fp32.t().contiguous()).to(a.dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        db = (
            _C.det_gemm_rowwise_fwd_fp32(a_fp32.t().contiguous(), grad_fp32).to(b.dtype)
            if ctx.needs_input_grad[1]
            else None
        )
        record_backward(
            "det_gemm",
            kernel_id=("rl_engine._C.det_gemm_rowwise_fwd_fp32"),
            impl="cuda_rowwise_fp32_accum_det_gemm",
            family="cuda",
        )
        return da, db


class DetGemmOp:
    """Hand-written batch-invariant GEMM. a:[M,K] bf16, b:[K,N] bf16 -> [M,N] bf16."""

    def __init__(self):
        self.has_hardware_op = False
        backend = det_gemm_backend()
        if backend == _CUBLASLT_BACKEND:
            _configure_cublaslt_nosplitk()
        if not (_EXT_AVAILABLE and hasattr(_C, "det_gemm_fwd")):
            raise RuntimeError(
                "strict RL-Kernel GEMM requires the compiled CUDA extension; "
                "refusing non-strict fallback"
            )
        if backend == _SM90_BACKEND and os.getenv(
            "RL_KERNEL_DET_GEMM_SM90_ONLY", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        } and not (
            hasattr(_C, "det_gemm_sm90_compiled") and _C.det_gemm_sm90_compiled()
        ):
            raise RuntimeError(
                "RL_KERNEL_DET_GEMM_SM90_ONLY=1 requires an extension built with "
                "KERNEL_ALIGN_DET_GEMM_SM90=1; refusing non-strict fallback"
            )
        self.op = _C.det_gemm_fwd
        self.has_hardware_op = True
        _report_strict_route_once()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
        if not self.has_hardware_op:
            raise RuntimeError(
                "DetGemmOp: compiled _C.det_gemm kernel unavailable; no "
                "batch-invariant fallback exists. Build the extension first."
            )
        return _DetGemmFn.apply(a.contiguous(), b.contiguous(), False)

    def linear(self, a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Apply a native [N,K] linear weight without materializing weight.T."""
        assert a.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
        required = ("det_gemm_fwd_weight", "det_gemm_da_weight", "det_gemm_dw")
        if not self.has_hardware_op or any(not hasattr(_C, name) for name in required):
            raise RuntimeError(
                "DetGemmOp.linear requires the rebuilt native-weight CUDA extension"
            )
        return _DetLinearFn.apply(a.contiguous(), weight.contiguous())

    def forward_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
        if not self.has_hardware_op:
            raise RuntimeError("DetGemmOp: compiled CUDA extension unavailable")
        return _DetGemmFn.apply(a.contiguous(), b.contiguous(), True)

    def forward_accum_fp32(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dtype not in (torch.bfloat16, torch.float32) or b.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise TypeError("FP32-accumulation GEMM requires BF16 or FP32 inputs")
        assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
        return _DetGemmAccumFn.apply(a.contiguous(), b.contiguous())

    def parameter_vjp_contributions_fp32(self, *, a, b, grad_output):
        del b
        rows_a = a.float()
        rows_g = grad_output.float()
        return {"b": rows_a[:, :, None] * rows_g[:, None, :]}


def deterministic_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Functional entry. a:[M,K] bf16, b:[K,N] bf16 -> [M,N] bf16."""
    return _DetGemmFn.apply(a, b, False)
