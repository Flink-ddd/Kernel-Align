# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariant deterministic GEMM, CUDA path (WS1 #146).

Hand-written kernel (csrc/cuda/gemm/det_gemm_kernel.cu): fixed K-accumulation
order, FP32 accumulation, no split-K. A row's output is invariant to batch size,
chunked-prefill, and padding. No PyTorch fallback -- a generic matmul (cuBLAS)
would silently break invariance (see NativeGemmOp). Tensor-parallel GEMM is WS2.
"""
import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


class _DetGemmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, output_fp32=False):
        ctx.save_for_backward(a, b)
        if output_fp32:
            if not hasattr(_C, "det_gemm_fwd_fp32"):
                raise RuntimeError("FP32 deterministic GEMM output requires the rebuilt extension")
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
        if _EXT_AVAILABLE and hasattr(_C, "det_gemm_fwd"):
            self.op = _C.det_gemm_fwd
            self.has_hardware_op = True
            logger.info("Successfully linked to RL-Kernel _C.det_gemm_fwd.")
        else:
            logger.warning(
                "RL-Kernel _C.det_gemm_fwd unavailable; DetGemmOp requires the "
                "compiled CUDA extension and has no batch-invariant fallback."
            )

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA device"
        if not self.has_hardware_op:
            raise RuntimeError(
                "DetGemmOp: compiled _C.det_gemm kernel unavailable; no "
                "batch-invariant fallback exists. Build the extension first."
            )
        return _DetGemmFn.apply(a.contiguous(), b.contiguous(), False)

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
