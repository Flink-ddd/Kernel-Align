# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton RoPE kernel (GPT-NeoX / HF rotate-half), matching NativeRoPEOp.

For a token at absolute position ``p`` and head_dim index ``d`` (half = D // 2)::

    inv_freq[i] = theta ** (-i / half)              i in [0, half)
    angle       = p * inv_freq[d % half]
    out[d<half]  = x[d] * cos(angle) - x[d+half] * sin(angle)
    out[d>=half] = x[d] * cos(angle) + x[d-half] * sin(angle)

cos/sin are built in fp32 with the *exact* reference math (``theta ** x``) so the
fp32 path stays bit-close to the gold even at large positions where cos/sin are
numerically sensitive; the Triton kernel does the elementwise rotation in fp32
and rounds back to the input dtype on store.

RoPE is a per-position orthogonal rotation, so the input gradient is the same
rotation with the sine negated::

    grad_x = grad_out * cos - rotate_half(grad_out) * sin

which the same kernel produces when called with ``sin_sign = -1``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_kernel(
    x_ptr,  # [n_rows, D] flattened input
    cos_ptr,  # [S, HALF] fp32 cosine cache
    sin_ptr,  # [S, HALF] fp32 sine cache
    out_ptr,  # [n_rows, D] output
    n_rows,
    S,
    SIN_SIGN: tl.constexpr,  # +1.0 forward, -1.0 backward
    HALF: tl.constexpr,  # D // 2, power-of-two block width
    stride_row,
    stride_d,
):
    """One program per row (a single [B, H, S] token vector of width D)."""
    row = tl.program_id(0)
    if row >= n_rows:
        return

    # Row layout is [..., S, D] contiguous, so the sequence index is row % S.
    seq_idx = row % S
    d = tl.arange(0, HALF)
    cos = tl.load(cos_ptr + seq_idx * HALF + d)
    sin = tl.load(sin_ptr + seq_idx * HALF + d) * SIN_SIGN

    base = row * stride_row
    x1 = tl.load(x_ptr + base + d * stride_d).to(tl.float32)
    x2 = tl.load(x_ptr + base + (d + HALF) * stride_d).to(tl.float32)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + base + d * stride_d, out1.to(out_dtype))
    tl.store(out_ptr + base + (d + HALF) * stride_d, out2.to(out_dtype))


def _build_cos_sin(positions: Tensor, half: int, theta: float, device: torch.device):
    """fp32 cos/sin caches of shape [S, half], identical math to NativeRoPEOp."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = positions.to(device=device, dtype=torch.float32).reshape(-1, 1)
    freqs = pos * inv_freq  # [S, half]
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def _launch_rope(x: Tensor, cos: Tensor, sin: Tensor, S: int, sin_sign: float) -> Tensor:
    D = x.shape[-1]
    half = D // 2
    x_2d = x.contiguous().reshape(-1, D)
    n_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d)
    grid = (n_rows,)
    _rope_kernel[grid](
        x_2d,
        cos,
        sin,
        out,
        n_rows,
        S,
        SIN_SIGN=float(sin_sign),
        HALF=half,
        stride_row=x_2d.stride(0),
        stride_d=x_2d.stride(1),
    )
    return out.reshape(x.shape)


class _RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, positions: Tensor, theta: float) -> Tensor:
        D = x.shape[-1]
        if D % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {D}")
        if positions.dim() != 1:
            raise NotImplementedError(
                "Triton RoPE currently supports 1-D positions [S] (shared across batch)."
            )
        S = positions.shape[0]
        n_rows = x.numel() // D
        if n_rows % S != 0:
            raise ValueError(
                f"row count {n_rows} not divisible by seq length {S}; "
                "expected a [..., S, D] contiguous layout."
            )
        cos, sin = _build_cos_sin(positions, D // 2, float(theta), x.device)
        ctx.save_for_backward(cos, sin)
        ctx.seq_len = S
        return _launch_rope(x, cos, sin, S, sin_sign=1.0)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        cos, sin = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            # Inverse rotation: same kernel with the sine negated.
            grad_x = _launch_rope(grad_out, cos, sin, ctx.seq_len, sin_sign=-1.0)
        # Inputs: x, positions, theta.
        return grad_x, None, None


class TritonRoPEOp:
    """Triton RoPE op (GPT-NeoX rotate-half), differentiable w.r.t. ``x``.

    Qwen3 defaults: theta=1e6, head_dim=128, full-dimension rotation. cos/sin are
    computed in fp32 from ``positions`` and ``theta`` (matching the reference) and
    the rotation runs in a Triton kernel -- no external cos/sin cache is accepted.
    """

    op_class = "elementwise"

    def __call__(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        return self.forward(x, positions, theta=theta)

    def forward(self, x: Tensor, positions: Tensor, *, theta: float = 1_000_000.0) -> Tensor:
        if x.device.type not in ("cuda", "hip", "xpu"):
            raise RuntimeError(f"TritonRoPEOp requires a GPU tensor, got device '{x.device}'.")
        return _RoPEFunction.apply(x, positions, theta)
