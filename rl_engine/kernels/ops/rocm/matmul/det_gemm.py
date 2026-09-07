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
_WEIGHT_TRANSPOSE_CACHE: dict[
    tuple[int, str, tuple[int, ...], torch.dtype], tuple[int, torch.Tensor]
] = {}
_WEIGHT_TRANSPOSE_CACHE_LIMIT = 256
_DIRECT_STAGING_BY_SLOT: dict[int, tuple[int, torch.Tensor]] = {}
_DIRECT_STAGING_SLOT_BY_HANDLE: dict[int, int] = {}


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


def _cached_weight_transpose(weight: torch.Tensor) -> torch.Tensor:
    """Reuse inference-only transposed weights until the parameter is updated."""

    # Training must retain the original autograd graph and therefore cannot
    # retain a detached transpose in a process-global cache.
    if torch.is_grad_enabled():
        return weight.t().contiguous()
    key = (weight.data_ptr(), str(weight.device), tuple(weight.shape), weight.dtype)
    version = int(getattr(weight, "_version", 0))
    cached = _WEIGHT_TRANSPOSE_CACHE.get(key)
    if cached is not None:
        if cached[0] != version:
            # Preserve the allocation captured by HIP Graph while refreshing
            # its contents after an in-place framework weight update.
            cached[1].copy_(weight.t())
            _WEIGHT_TRANSPOSE_CACHE[key] = (version, cached[1])
        return cached[1]
    transposed = weight.t().contiguous()
    if len(_WEIGHT_TRANSPOSE_CACHE) >= _WEIGHT_TRANSPOSE_CACHE_LIMIT:
        _WEIGHT_TRANSPOSE_CACHE.pop(next(iter(_WEIGHT_TRANSPOSE_CACHE)))
    _WEIGHT_TRANSPOSE_CACHE[key] = (version, transposed)
    return transposed


@torch.inference_mode()
def refresh_cached_weight_transposes(weights: object) -> int:
    """Refresh graph-captured transpose buffers after an IPC weight update."""

    refreshed = 0
    for weight in weights:
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        key = (weight.data_ptr(), str(weight.device), tuple(weight.shape), weight.dtype)
        cached = _WEIGHT_TRANSPOSE_CACHE.get(key)
        if cached is None:
            continue
        cached[1].copy_(weight.t())
        version = int(getattr(weight, "_version", 0))
        _WEIGHT_TRANSPOSE_CACHE[key] = (version, cached[1])
        refreshed += 1
    return refreshed


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
    return _triton_tree_gemm(a, _cached_weight_transpose(weight), out=out)


@torch.library.custom_op("rl_kernel::rocm_det_gemm_linear_inference", mutates_args=())
def _det_gemm_linear_inference(
    a: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return det_gemm_linear(a, weight)


@_det_gemm_linear_inference.register_fake
def _det_gemm_linear_inference_fake(
    a: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return a.new_empty((a.shape[0], weight.shape[0]))


@torch.library.custom_op(
    "rl_kernel::rocm_det_gemm_linear_inference_out",
    mutates_args={"out"},
)
def _det_gemm_linear_inference_out(
    a: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> None:
    det_gemm_linear(a, weight, out=out)


@_det_gemm_linear_inference_out.register_fake
def _det_gemm_linear_inference_out_fake(
    a: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> None:
    del a, weight, out


@torch.library.custom_op(
    "rl_kernel::rocm_det_gemm_linear_all_reduce_inference",
    mutates_args=(),
)
def _det_gemm_linear_all_reduce_inference(
    a: torch.Tensor,
    weight: torch.Tensor,
    collective_handle: int,
) -> torch.Tensor:
    """Run row-parallel inference without exposing shape dispatch to Dynamo."""

    from rl_engine import _C

    binding = _DIRECT_STAGING_BY_SLOT.get(collective_handle)
    if binding is None:
        raise RuntimeError("strict ROCm row-parallel staging handle is not registered")
    runtime_handle, staging = binding
    if a.size(0) <= staging.size(0):
        direct_output = staging.narrow(0, 0, a.size(0))
        _C.deterministic_collective_rocm_ipc_prepare_staged(
            runtime_handle,
            direct_output,
        )
        det_gemm_linear(a, weight, out=direct_output)
        output = torch.empty_like(direct_output)
        _C.deterministic_collective_rocm_ipc_all_reduce_staged(
            runtime_handle,
            direct_output,
            output,
        )
        return output

    output = det_gemm_linear(a, weight)
    _C.deterministic_collective_rocm_ipc_all_reduce_input(
        runtime_handle,
        output,
        output,
    )
    return output


@_det_gemm_linear_all_reduce_inference.register_fake
def _det_gemm_linear_all_reduce_inference_fake(
    a: torch.Tensor,
    weight: torch.Tensor,
    collective_handle: int,
) -> torch.Tensor:
    del collective_handle
    return a.new_empty((a.shape[0], weight.shape[0]))


def register_det_gemm_all_reduce_staging(
    collective_handle: int,
    staging: torch.Tensor,
) -> int:
    if collective_handle <= 0:
        raise ValueError("collective_handle must be positive")
    slot = _DIRECT_STAGING_SLOT_BY_HANDLE.get(collective_handle)
    if slot is None:
        # Dynamo persists scalar custom-op arguments in its cross-process AOT
        # cache. Use registration order as the stable graph identity and resolve
        # the process-local C++ handle only when the custom op executes.
        slot = len(_DIRECT_STAGING_SLOT_BY_HANDLE) + 1
        _DIRECT_STAGING_SLOT_BY_HANDLE[collective_handle] = slot
    _DIRECT_STAGING_BY_SLOT[slot] = (collective_handle, staging)
    return slot


def det_gemm_linear_all_reduce_inference(
    a: torch.Tensor,
    weight: torch.Tensor,
    *,
    collective_handle: int,
) -> torch.Tensor:
    return _det_gemm_linear_all_reduce_inference(
        a,
        weight,
        collective_handle,
    )


def prepare_det_gemm_linear_weight(
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prepare a stable contiguous ``[K,N]`` inference weight.

    ROCm's strict tree kernel consumes its RHS in ``[K,N]`` layout.  vLLM
    stores an unquantized linear weight as ``[N,K]`` instead, so materializing
    the transpose in every decode call is especially expensive for the
    LM-head.  The caller owns freshness: create or refresh this tensor from a
    verified model load/update lifecycle, never from a version-counter guess.
    """

    if weight.dim() != 2:
        raise ValueError("prepared deterministic linear weights must be 2-D")
    if weight.dtype != torch.bfloat16:
        raise TypeError("prepared deterministic linear weights must be BF16")
    if not weight.is_cuda:
        raise RuntimeError("prepared deterministic linear weights must be on ROCm")
    if not weight.is_contiguous():
        raise ValueError("source deterministic linear weights must be contiguous")

    expected_shape = (weight.size(1), weight.size(0))
    if out is None:
        # vLLM invokes post-load hooks from inference mode.  Allocate an
        # ordinary tensor so later hot-weight refreshes may update it in place.
        with torch.inference_mode(False), torch.no_grad():
            out = torch.empty(
                expected_shape,
                dtype=weight.dtype,
                device=weight.device,
            )
    else:
        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"prepared deterministic linear weight must have shape "
                f"{expected_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != weight.dtype:
            raise TypeError("prepared deterministic linear weight dtype must match its source")
        if out.device != weight.device:
            raise RuntimeError("prepared deterministic linear weight device must match its source")
        if not out.is_contiguous():
            raise ValueError("prepared deterministic linear weight must be contiguous")
        if out.requires_grad:
            raise ValueError("prepared deterministic linear weight must not require gradients")
        if torch._C._overlaps(out, weight):
            raise ValueError("prepared deterministic linear weight must not alias its source")

    with torch.no_grad():
        out.copy_(weight.t())
    return out


def det_gemm_linear_prepared(
    a: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a lifecycle-managed contiguous ``[K,N]`` inference weight."""

    if torch.is_grad_enabled() and (a.requires_grad or weight_t.requires_grad):
        raise RuntimeError("prepared deterministic linear GEMM is inference-only")
    if not weight_t.is_contiguous():
        raise ValueError("prepared deterministic linear weight must be contiguous")
    if out is not None and (torch._C._overlaps(out, a) or torch._C._overlaps(out, weight_t)):
        raise ValueError("prepared deterministic linear output must not alias its inputs")
    return _triton_tree_gemm(a, weight_t, out=out)


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
        da = det_gemm_linear_input_gradient(grad_out, weight) if ctx.needs_input_grad[0] else None
        dweight = det_gemm_linear_weight_gradient(a, grad_out) if ctx.needs_input_grad[1] else None
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
            _det_gemm_linear_inference_out(a, weight, out)
            return out
        if not torch.is_grad_enabled() or not (a.requires_grad or weight.requires_grad):
            return _det_gemm_linear_inference(a, weight)
        return _DetLinearFn.apply(a, weight)

    def linear_prepared(
        self,
        a: torch.Tensor,
        weight_t: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a post-load prepared RHS without a hot-path transpose."""

        assert a.dtype == torch.bfloat16 and weight_t.dtype == torch.bfloat16, "BF16 only"
        assert a.is_cuda and weight_t.is_cuda, "Inputs must be on ROCm device"
        return det_gemm_linear_prepared(a.contiguous(), weight_t, out=out)

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
    "det_gemm_linear_prepared",
    "det_gemm_linear_input_gradient",
    "det_gemm_linear_weight_gradient",
    "deterministic_gemm",
    "prepare_det_gemm_linear_weight",
    "det_gemm_linear_all_reduce_inference",
    "register_det_gemm_all_reduce_staging",
    "refresh_cached_weight_transposes",
]
