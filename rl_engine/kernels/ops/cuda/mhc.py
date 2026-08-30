# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


def _require_mhc_extension() -> None:
    required = ("mhc_pre_h_aggregate", "mhc_pre_h_aggregate_backward")
    if not _EXT_AVAILABLE or _C is None or not all(hasattr(_C, name) for name in required):
        raise RuntimeError("MHC H Aggregate CUDA symbols are unavailable; rebuild rl_engine._C")


class _MHCPreHAggregateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        residual = residual.contiguous()
        pre = pre.contiguous()
        output = _C.mhc_pre_h_aggregate(residual, pre)
        ctx.save_for_backward(residual, pre)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        residual, pre = ctx.saved_tensors
        grad_residual = grad_pre = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grads = _C.mhc_pre_h_aggregate_backward(grad_output.contiguous(), residual, pre)
            if ctx.needs_input_grad[0]:
                grad_residual = grads[0]
            if ctx.needs_input_grad[1]:
                grad_pre = grads[1]
        record_backward(
            "mhc_pre_h_aggregate",
            kernel_id="rl_engine._C.mhc_pre_h_aggregate_backward",
            impl="cuda_fixed_tree_mhc_pre_h_aggregate_backward",
            family="cuda",
        )
        return grad_residual, grad_pre


class MHCPreHAggregateCudaOp:
    """Autograd-enabled CUDA MHC weighted collapse."""

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "cuda_fixed_tree_mhc_pre_h_aggregate_backward"

    def __init__(self) -> None:
        _require_mhc_extension()

    def __call__(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self.forward(residual, pre)

    def forward(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return _MHCPreHAggregateFunction.apply(residual, pre)
