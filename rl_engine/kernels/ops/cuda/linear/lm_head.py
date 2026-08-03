# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.linear.lm_head import NativeLMHeadOp
from rl_engine.utils.logger import logger

_SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _is_hopper(device: torch.device) -> bool:
    try:
        return torch.cuda.get_device_capability(device)[0] == 9
    except Exception:
        return False


def _can_use_det_gemm_backward(hidden: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        hidden.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and _EXT_AVAILABLE
        and hasattr(_C, "det_gemm_da")
        and hasattr(_C, "det_gemm_db")
    )


class _SM90LMHeadFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        output_fp32: bool,
    ):
        bias_to_save = (
            bias if bias is not None else torch.empty(0, device=hidden.device, dtype=hidden.dtype)
        )
        ctx.save_for_backward(hidden, weight, bias_to_save)
        ctx.has_bias = bias is not None
        ctx.output_fp32 = bool(output_fp32)
        weight_c = weight.contiguous()
        if output_fp32:
            return _C.lm_head_sm90_forward_fp32(hidden, weight_c, bias)
        return _C.lm_head_sm90_forward(hidden, weight_c, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, bias = ctx.saved_tensors
        grad_hidden = grad_weight = grad_bias = None

        hidden_2d = hidden.reshape(-1, hidden.size(-1))
        grad_2d = grad_output.reshape(-1, weight.size(0)).float()
        hidden_f = hidden_2d.float()
        weight_f = weight.float()
        needs_projection_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        use_det_gemm = (
            hidden_2d.size(0) > 0
            and needs_projection_grad
            and _can_use_det_gemm_backward(hidden, weight)
        )
        if (
            hidden_2d.size(0) > 0
            and needs_projection_grad
            and hidden.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and not use_det_gemm
        ):
            raise RuntimeError(
                "SM90LMHeadOp.backward requires _C.det_gemm_da/db for bf16 "
                "batch-invariant gradients."
            )

        if ctx.needs_input_grad[0]:
            if use_det_gemm:
                grad_hidden = _C.det_gemm_da(
                    grad_2d.to(torch.bfloat16).contiguous(),
                    weight.t().contiguous(),
                )
            else:
                grad_hidden = grad_2d.matmul(weight_f)
            grad_hidden = grad_hidden.reshape_as(hidden).to(hidden.dtype)
        if ctx.needs_input_grad[1]:
            if use_det_gemm:
                grad_weight = _C.det_gemm_db(
                    hidden_2d.contiguous(),
                    grad_2d.to(torch.bfloat16).contiguous(),
                ).t()
            else:
                grad_weight = grad_2d.transpose(0, 1).matmul(hidden_f)
            grad_weight = grad_weight.contiguous().to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_2d.sum(0).to(bias.dtype)

        return grad_hidden, grad_weight, grad_bias, None


class SM90LMHeadOp:
    """Single-card SM90 batch-invariant CUDA LM-head op.

    The CUDA forward launches one CTA per output logit and performs the full K
    reduction in that CTA. There is no Split-K and no cuBLAS algorithm selection
    in the forward path, so a row's logits do not depend on batch layout.
    """

    op_class = "reduction"
    is_batch_invariant = True

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "lm_head_sm90_forward"):
            raise RuntimeError(
                "lm_head_sm90_forward is not compiled into the extension. "
                "Rebuild on Hopper with KERNEL_ALIGN_FORCE_SM90=1."
            )
        self._fallback = NativeLMHeadOp()
        logger.info("Successfully linked to precompiled _C.lm_head_sm90_forward kernel.")

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(hidden, weight, bias=bias)

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._can_use_sm90(hidden, weight, bias):
            return self._fallback.forward(hidden, weight, bias=bias)
        return _SM90LMHeadFunction.apply(hidden, weight, bias, False)

    def forward_fp32(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._can_use_sm90(hidden, weight, bias):
            return self._fallback.forward_fp32(hidden, weight, bias=bias)
        return _SM90LMHeadFunction.apply(hidden, weight, bias, True)

    @staticmethod
    def _can_use_sm90(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> bool:
        bias_ok = bias is None or (
            bias.is_cuda
            and bias.device == hidden.device
            and bias.dim() == 1
            and bias.dtype in _SUPPORTED_DTYPES
        )
        return (
            hidden.is_cuda
            and weight.is_cuda
            and hidden.device == weight.device
            and _is_hopper(hidden.device)
            and hidden.dim() >= 2
            and weight.dim() == 2
            and hidden.size(-1) == weight.size(1)
            and hidden.dtype in _SUPPORTED_DTYPES
            and weight.dtype in _SUPPORTED_DTYPES
            and bias_ok
        )
