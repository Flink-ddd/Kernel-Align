"""Platform-selected matrix multiplication operators."""

from .det_gemm import DetGemmOp, deterministic_gemm

__all__ = ["DetGemmOp", "deterministic_gemm"]
