# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Single-card batch-invariant fused linear log-probability on SM90."""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_BIAS_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_HIDDEN_TILE = 32


def _validate_target_range(target_ids: torch.Tensor, vocab_size: int) -> None:
    # Compare in int64 so Python scalars such as a large vocabulary size do not
    # overflow narrow integer tensors during validation.
    target_i64 = target_ids.to(dtype=torch.int64)
    invalid = (target_i64 < 0) | (target_i64 >= vocab_size)
    if bool(invalid.any()):
        invalid_targets = target_i64[invalid]
        target_min = int(invalid_targets.min())
        target_max = int(invalid_targets.max())
        raise ValueError(
            f"target_ids must be in [0, {vocab_size}), got invalid range "
            f"[{target_min}, {target_max}]"
        )


def _validate_inputs(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    validate_targets: bool = False,
) -> None:
    if hidden.dim() < 2:
        raise ValueError(f"hidden must be at least 2-D [*lead, D], got {tuple(hidden.shape)}")
    if lm_head_weight.dim() != 2:
        raise ValueError(f"lm_head_weight must be 2-D [V, D], got {tuple(lm_head_weight.shape)}")
    if hidden.shape[:-1] != target_ids.shape:
        raise ValueError(
            f"hidden leading shape {tuple(hidden.shape[:-1])} must match "
            f"target_ids shape {tuple(target_ids.shape)}"
        )
    if lm_head_weight.size(1) != hidden.size(-1):
        raise ValueError(
            f"hidden dim {hidden.size(-1)} must match weight dim {lm_head_weight.size(1)}"
        )
    if hidden.size(-1) == 0:
        raise ValueError("hidden dimension must be positive")
    if target_ids.numel() == 0:
        raise ValueError("hidden must contain at least one token row")
    if lm_head_weight.size(0) == 0:
        raise ValueError("lm_head_weight must contain at least one vocabulary row")
    if not hidden.is_cuda or not lm_head_weight.is_cuda or not target_ids.is_cuda:
        raise ValueError("hidden, lm_head_weight, and target_ids must be CUDA tensors")
    if lm_head_weight.device != hidden.device or target_ids.device != hidden.device:
        raise ValueError("hidden, lm_head_weight, and target_ids must be on the same CUDA device")
    if hidden.dtype != torch.bfloat16 or lm_head_weight.dtype != torch.bfloat16:
        raise TypeError("batch_invariant_linear_logp_sm90 supports bf16 hidden and weight only")
    if target_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"target_ids must have an integer dtype, got {target_ids.dtype}")
    if hidden.size(-1) % _HIDDEN_TILE != 0:
        raise ValueError(f"hidden dim must be a multiple of {_HIDDEN_TILE}, got {hidden.size(-1)}")
    if torch.cuda.get_device_capability(hidden.device)[0] != 9:
        raise RuntimeError("batch_invariant_linear_logp_sm90 requires an SM90 Hopper GPU")

    if bias is not None:
        if bias.dim() != 1 or bias.numel() != lm_head_weight.size(0):
            raise ValueError(
                f"bias must have shape ({lm_head_weight.size(0)},), got {tuple(bias.shape)}"
            )
        if not bias.is_cuda or bias.device != hidden.device:
            raise ValueError("bias must be a CUDA tensor on the same device as hidden")
        if bias.dtype not in _BIAS_DTYPES:
            raise TypeError(f"bias must have dtype fp16, bf16, or fp32, got {bias.dtype}")

    if torch.is_grad_enabled() and (
        hidden.requires_grad
        or lm_head_weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        raise RuntimeError(
            "batch_invariant_linear_logp_sm90 is forward-only; call it under "
            "torch.no_grad() or use a differentiable linear_logp backend"
        )

    if validate_targets:
        _validate_target_range(target_ids, lm_head_weight.size(0))


class BatchInvariantLinearLogpSM90Op:
    """Fused ``log_softmax(hidden @ weight.T + bias)[target]`` for Hopper.

    The operator never materializes ``[N, V]`` logits. Its SM90 launch uses a
    fixed D traversal and a split-V schedule derived only from V, so a row's
    float32 output is bitwise invariant to batch size, row position, neighboring
    rows, and batch-dimension chunking. Backward and tensor parallelism are
    intentionally outside this first single-card contract.
    """

    op_class = "logprob"
    is_batch_invariant = True
    supports_backward = False

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "batch_invariant_linear_logp_sm90"):
            raise RuntimeError(
                "batch_invariant_linear_logp_sm90 is not compiled into the extension. "
                "Rebuild on Hopper with KERNEL_ALIGN_FORCE_SM90=1."
            )
        logger.info("Linked _C.batch_invariant_linear_logp_sm90.")

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        validate: bool = False,
    ) -> torch.Tensor:
        return self.forward(hidden, lm_head_weight, target_ids, bias, validate=validate)

    def forward(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        validate: bool = False,
    ) -> torch.Tensor:
        _validate_inputs(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            validate_targets=validate,
        )

        lead_shape = hidden.shape[:-1]
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight_2d = lm_head_weight.contiguous()
        target_1d = target_ids.reshape(-1)

        logp, _lse = _C.batch_invariant_linear_logp_sm90(
            hidden_2d,
            weight_2d,
            target_1d,
            bias,
        )
        return logp.reshape(lead_shape)
