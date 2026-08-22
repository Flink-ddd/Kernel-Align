# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logical-row canonical CUDA LM-head autograd for WS1."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.canonical_backward import active_session


class _CanonicalCudaLMHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, logical_keys, parameter_id):
        session = active_session()
        if session is None:
            raise RuntimeError("canonical LM-head requires an active backward session")
        ctx.save_for_backward(hidden, weight)
        ctx.session = session
        ctx.parameter_id = str(parameter_id)
        ctx.slot = session.register(ctx.parameter_id, logical_keys)
        return _C.lm_head_sm90_forward_fp32(hidden, weight.contiguous(), None)

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight = ctx.saved_tensors
        grad_rows = grad_out.reshape(-1, grad_out.shape[-1]).float()
        hidden_rows = hidden.reshape(-1, hidden.shape[-1])
        grad_hidden = (
            _C.det_gemm_rowwise_fwd_fp32(grad_rows, weight.float())
            .reshape_as(hidden)
            .to(hidden.dtype)
        )

        def reducer(rows, grads):
            return _C.det_gemm_rowwise_fwd_fp32(grads.float().t().contiguous(), rows.float()).to(
                weight.dtype
            )

        grad_weight = ctx.session.submit_linear(
            ctx.parameter_id, ctx.slot, hidden_rows, grad_rows, reducer
        )
        return grad_hidden, grad_weight, None, None


def canonical_cuda_lm_head_fp32(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    logical_keys: torch.Tensor,
    *,
    parameter_id: str = "lm_head",
) -> torch.Tensor:
    return _CanonicalCudaLMHead.apply(hidden, weight, logical_keys, parameter_id)


__all__ = ["canonical_cuda_lm_head_fp32"]


class _CanonicalRowLMHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, logical_keys, parameter_id, forward_op, matmul_op):
        session = active_session()
        if session is None:
            raise RuntimeError("canonical LM-head requires an active backward session")
        with torch.no_grad():
            output = forward_op(hidden, weight, bias=None)
        ctx.save_for_backward(hidden, weight)
        ctx.session = session
        ctx.parameter_id = str(parameter_id)
        ctx.slot = session.register(ctx.parameter_id, logical_keys)
        ctx.matmul_op = matmul_op
        return output

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight = ctx.saved_tensors
        grad_rows = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        hidden_rows = hidden.reshape(-1, hidden.shape[-1]).contiguous()
        grad_hidden = (
            ctx.matmul_op(grad_rows.to(weight.dtype), weight).reshape_as(hidden).to(hidden.dtype)
        )

        def reducer(rows, grads):
            return ctx.matmul_op(
                grads.to(weight.dtype).t().contiguous(),
                rows.to(weight.dtype),
            ).to(weight.dtype)

        grad_weight = ctx.session.submit_linear(
            ctx.parameter_id, ctx.slot, hidden_rows, grad_rows, reducer
        )
        record_backward(
            "lm_head",
            kernel_id="rl_engine.kernels.ops.triton.matmul.det_gemm._triton_gemm",
            impl="triton_lm_head_canonical_rowfold",
            family="triton",
        )
        return grad_hidden, grad_weight, None, None, None, None


def canonical_row_lm_head(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    logical_keys: torch.Tensor,
    *,
    forward_op,
    matmul_op,
    parameter_id: str = "lm_head",
) -> torch.Tensor:
    return _CanonicalRowLMHead.apply(
        hidden, weight, logical_keys, parameter_id, forward_op, matmul_op
    )
