# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch


class NativeMHCPreHAggregateOp:
    """PyTorch reference for the four-stream MHC weighted collapse."""

    op_class = "reduction"

    def __call__(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self.forward(residual, pre)

    def forward(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self._compute(residual, pre).to(residual.dtype)

    def forward_fp32(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self._compute(residual, pre)

    @staticmethod
    def _compute(residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        if residual.dim() != 3 or residual.shape[1] != 4:
            raise ValueError("residual must have shape [num_tokens, 4, hidden_size]")
        if pre.shape != residual.shape[:2]:
            raise ValueError("pre must have shape [num_tokens, 4]")
        if residual.device != pre.device:
            raise RuntimeError("residual and pre must be on the same device")

        residual_fp32 = residual.float()
        pre_fp32 = pre.float()
        left = (
            pre_fp32[:, 0, None] * residual_fp32[:, 0] + pre_fp32[:, 1, None] * residual_fp32[:, 1]
        )
        right = (
            pre_fp32[:, 2, None] * residual_fp32[:, 2] + pre_fp32[:, 3, None] * residual_fp32[:, 3]
        )
        return left + right
