# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch


@torch.library.custom_op("rl_kernel::strict_rms_norm", mutates_args=())
def _strict_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Preserve PyTorch eager RMSNorm arithmetic across graph compilation."""

    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


@_strict_rms_norm.register_fake
def _strict_rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del weight, eps
    return torch.empty_like(x)


@torch.library.custom_op("rl_kernel::strict_add_rms_norm", mutates_args=())
def _strict_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve vLLM's eager residual-add and RMSNorm contract."""

    updated_residual = x + residual
    normalized = torch.nn.functional.rms_norm(
        updated_residual,
        (updated_residual.shape[-1],),
        weight,
        eps,
    )
    return normalized, updated_residual


@_strict_add_rms_norm.register_fake
def _strict_add_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del residual, weight, eps
    return torch.empty_like(x), torch.empty_like(x)


def strict_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    return _strict_rms_norm(x, weight, eps)


def strict_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _strict_add_rms_norm(x, residual, weight, eps)


class NativeRMSNormOp:
    """
    Pure Pytorch native RMSNorm reference
    out = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return self.forward(x, weight, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Canonical entry: accumulate in fp32, cast the result back to x.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._rms_norm(x, weight, eps=eps, output_dtype=x.dtype)

    def forward_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Ground-truth: accumulate in fp32 and force fp32 output."""
        return self._rms_norm(x, weight, eps=eps, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
            raise ValueError(
                f"weight must be 1-D of size x.shape[-1]={x.shape[-1]}, "
                f"got tuple(weight.shape)={tuple(weight.shape)}"
            )
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f * torch.rsqrt(var + eps)
        out = normed * weight.float()
        return out.to(output_dtype)
