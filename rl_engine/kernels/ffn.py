# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen3 FFN orchestration for the WS2 consistency and fast paths.

The module owns no process groups and launches no collectives. It consumes the
rank-local token tensor supplied by the distributed wrapper (after any SP
ownership conversion) and returns the local Down projection partial before the
wrapper's TP AllReduce or sequence-parallel ReduceScatter::

    gate   = GEMM(x, W_gate)
    up     = GEMM(x, W_up)
    hidden = SwiGLU(gate, up)
    output = GEMM(hidden, W_down)

For an unsharded Qwen3-8B FFN, ``intermediate_size`` is 12288.  Under TP=2
it is 6144, so the same orchestration object is reusable by the distributed
wrappers without hiding communication inside the FFN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

GemmCallable = Callable[[Tensor, Tensor], Tensor]
SwiGLUCallable = Callable[[Tensor, Tensor], Tensor]

QWEN3_8B_HIDDEN_SIZE = 4096
QWEN3_8B_INTERMEDIATE_SIZE = 12288
QWEN3_8B_TP2_INTERMEDIATE_SIZE = 6144


@dataclass(frozen=True)
class Qwen3FFNProvenance:
    """The selected execution path, recorded by tests and benchmarks."""

    path: str
    gemm_backend: str
    activation_backend: str


@dataclass(frozen=True)
class Qwen3FFNStages:
    """Observable FFN boundaries used by tolerance and backward validation."""

    gate: Tensor
    up: Tensor
    hidden: Tensor
    output: Tensor


def _validate_weights(gate_weight: Tensor, up_weight: Tensor, down_weight: Tensor) -> None:
    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("Qwen3 FFN weights must all be rank-2 GEMM matrices")
    if gate_weight.shape != up_weight.shape:
        raise ValueError(
            "gate and up weights must share [hidden, intermediate] shape, got "
            f"{tuple(gate_weight.shape)} and {tuple(up_weight.shape)}"
        )
    hidden_size, intermediate_size = gate_weight.shape
    if down_weight.shape != (intermediate_size, hidden_size):
        expected = (intermediate_size, hidden_size)
        actual = tuple(down_weight.shape)
        raise ValueError(
            "down weight must have [intermediate, hidden] shape " f"{expected}, got {actual}"
        )
    if not gate_weight.is_floating_point():
        dtype = gate_weight.dtype
        raise TypeError(f"FFN weights must be floating point, got {dtype}")
    if not (gate_weight.dtype == up_weight.dtype == down_weight.dtype):
        raise TypeError(
            "Qwen3 FFN weights must share dtype, got "
            f"{gate_weight.dtype}, {up_weight.dtype}, and {down_weight.dtype}"
        )
    if not (gate_weight.device == up_weight.device == down_weight.device):
        devices = (gate_weight.device, up_weight.device, down_weight.device)
        raise RuntimeError(f"FFN weights must share device, got {devices}")


def _make_parameter(value: Tensor, *, trainable: bool) -> nn.Parameter:
    if isinstance(value, nn.Parameter) and value.requires_grad == trainable:
        return value
    return nn.Parameter(value.detach(), requires_grad=trainable)


class Qwen3FFN(nn.Module):
    """Backend-agnostic Qwen3 FFN over explicit GEMM-layout weights.

    Weight layouts follow ``A[M,K] @ B[K,N]`` instead of ``nn.Linear``'s
    transposed storage:

    - ``gate_weight`` and ``up_weight``: ``[hidden, intermediate_local]``
    - ``down_weight``: ``[intermediate_local, hidden]``

    The leading input dimensions are flattened into the GEMM row dimension and
    restored on output.  No bias is present in the Qwen3 MLP contract.
    """

    def __init__(
        self,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        *,
        gemm_op: GemmCallable,
        swiglu_op: SwiGLUCallable,
        provenance: Qwen3FFNProvenance,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        _validate_weights(gate_weight, up_weight, down_weight)
        self.gate_weight = _make_parameter(gate_weight, trainable=trainable)
        self.up_weight = _make_parameter(up_weight, trainable=trainable)
        self.down_weight = _make_parameter(down_weight, trainable=trainable)
        self.gemm_op = gemm_op
        self.swiglu_op = swiglu_op
        self.provenance = provenance

    @property
    def hidden_size(self) -> int:
        return int(self.gate_weight.shape[0])

    @property
    def intermediate_size(self) -> int:
        """Return local intermediate width: 12288 full or 6144 at TP=2."""

        return int(self.gate_weight.shape[1])

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim < 2:
            shape = tuple(x.shape)
            raise ValueError(f"FFN input must be rank 2+, got {shape}")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Qwen3FFN input last dimension must be {self.hidden_size}, "
                f"got shape {tuple(x.shape)}"
            )
        if x.numel() == 0:
            raise ValueError("Qwen3FFN does not support an empty token axis")
        if x.dtype != self.gate_weight.dtype:
            raise TypeError(
                f"Qwen3FFN input and weights must share dtype, got {x.dtype} "
                f"and {self.gate_weight.dtype}"
            )
        if x.device != self.gate_weight.device:
            weight_device = self.gate_weight.device
            raise RuntimeError(
                f"FFN input/weights must share device, got " f"{x.device} and {weight_device}"
            )

    def forward_with_stages(self, x: Tensor) -> Qwen3FFNStages:
        """Run the FFN and expose every arithmetic boundary for validation."""

        self._validate_input(x)
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.hidden_size).contiguous()
        intermediate_shape = (*leading_shape, self.intermediate_size)
        output_shape = (*leading_shape, self.hidden_size)

        gate = self.gemm_op(x_2d, self.gate_weight).reshape(intermediate_shape)
        up = self.gemm_op(x_2d, self.up_weight).reshape(intermediate_shape)
        hidden = self.swiglu_op(gate, up)
        hidden_2d = hidden.reshape(-1, self.intermediate_size).contiguous()
        output_2d = self.gemm_op(hidden_2d, self.down_weight)

        return Qwen3FFNStages(
            gate=gate,
            up=up,
            hidden=hidden,
            output=output_2d.reshape(output_shape),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_stages(x).output


def _fast_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Framework-native fast path without an invariance claim."""

    return F.silu(gate) * up


def _resolve_ops(
    path: str, backend: str
) -> tuple[GemmCallable, SwiGLUCallable, Qwen3FFNProvenance]:
    normalized_path = path.strip().lower().replace("-", "_")
    normalized_backend = backend.strip().lower().replace("-", "_")

    if normalized_path == "fast":
        if normalized_backend not in {"pytorch", "torch"}:
            raise ValueError(
                "the fast FFN path currently requires backend='pytorch'; " f"got {backend!r}"
            )
        return (
            torch.matmul,
            _fast_swiglu,
            Qwen3FFNProvenance(
                path="fast",
                gemm_backend="pytorch.matmul",
                activation_backend="torch.nn.functional.silu",
            ),
        )

    if normalized_path != "consistent":
        expected = "'consistent' or 'fast'"
        raise ValueError(f"unsupported FFN path {path!r}; expected {expected}")

    if normalized_backend == "cuda":
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
        from rl_engine.kernels.ops.cuda.activation.swiglu import SwiGLUCudaOp
        from rl_engine.kernels.ops.cuda.matmul.det_gemm import DetGemmOp

        required_symbols = (
            "det_gemm_fwd",
            "det_gemm_da",
            "det_gemm_db",
            "swiglu_forward",
            "swiglu_backward",
        )
        missing_symbols = (
            required_symbols
            if not _EXT_AVAILABLE or _C is None
            else tuple(symbol for symbol in required_symbols if not hasattr(_C, symbol))
        )
        if missing_symbols:
            missing = ", ".join(missing_symbols)
            raise RuntimeError(
                "the CUDA consistent FFN backend requires the compiled "
                f"forward/backward symbols; missing: {missing}"
            )

        gemm = DetGemmOp()
        activation = SwiGLUCudaOp()
        return (
            gemm,
            activation,
            Qwen3FFNProvenance(
                path="consistent",
                gemm_backend="cuda.det_gemm",
                activation_backend="cuda.swiglu",
            ),
        )

    if normalized_backend == "triton":
        try:
            from rl_engine.kernels.ops.triton.activation.swiglu import TritonSwiGLUOp
            from rl_engine.kernels.ops.triton.matmul.det_gemm import TritonDetGemmOp
        except ImportError as error:
            raise RuntimeError("the Triton consistent FFN backend is unavailable") from error

        gemm = TritonDetGemmOp()
        activation = TritonSwiGLUOp()
        return (
            gemm,
            activation,
            Qwen3FFNProvenance(
                path="consistent",
                gemm_backend="triton.det_gemm",
                activation_backend="triton.swiglu",
            ),
        )

    raise ValueError(
        "the consistent FFN path requires backend='cuda' or backend='triton'; " f"got {backend!r}"
    )


def build_qwen3_ffn(
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    path: str,
    backend: str,
    trainable: bool = True,
) -> Qwen3FFN:
    """Construct an explicit consistent or fast Qwen3 FFN implementation.

    Backend resolution is fail-closed.  A requested deterministic CUDA/Triton
    implementation is never replaced by ``torch.matmul`` implicitly.
    """

    _validate_weights(gate_weight, up_weight, down_weight)
    normalized_path = path.strip().lower().replace("-", "_")
    normalized_backend = backend.strip().lower().replace("-", "_")
    valid_pair = (normalized_path, normalized_backend) in {
        ("consistent", "cuda"),
        ("consistent", "triton"),
        ("fast", "pytorch"),
        ("fast", "torch"),
    }
    if not valid_pair:
        # _resolve_ops owns the stable, user-facing error messages.
        _resolve_ops(path, backend)

    if normalized_path == "consistent":
        if gate_weight.dtype != torch.bfloat16:
            raise TypeError(
                "the consistent Qwen3 FFN path requires BF16 weights; " f"got {gate_weight.dtype}"
            )
        if gate_weight.device.type != "cuda":
            raise RuntimeError(
                "the consistent Qwen3 FFN path requires CUDA SM90 weights; "
                f"got {gate_weight.device}"
            )
        capability = torch.cuda.get_device_capability(gate_weight.device)
        if capability[0] != 9:
            raise RuntimeError(
                "the consistent Qwen3 FFN path targets SM90; "
                f"got SM{capability[0]}{capability[1]}"
            )

    gemm_op, swiglu_op, provenance = _resolve_ops(path, backend)
    return Qwen3FFN(
        gate_weight,
        up_weight,
        down_weight,
        gemm_op=gemm_op,
        swiglu_op=swiglu_op,
        provenance=provenance,
        trainable=trainable,
    )


def qwen3_ffn_fp32_reference(
    x: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Qwen3FFNStages:
    """Uninterrupted FP32 reference over the exact quantized input values."""

    _validate_weights(gate_weight, up_weight, down_weight)
    if x.ndim < 2 or x.shape[-1] != gate_weight.shape[0]:
        raise ValueError(
            f"reference input must end in hidden size {gate_weight.shape[0]}, "
            f"got {tuple(x.shape)}"
        )
    if x.device != gate_weight.device:
        raise RuntimeError("reference input and weights must share device")

    old_tf32: Any = None
    if x.device.type == "cuda":
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        leading_shape = x.shape[:-1]
        hidden_size, intermediate_size = gate_weight.shape
        x_fp32 = x.float().reshape(-1, hidden_size)
        intermediate_shape = (*leading_shape, intermediate_size)
        gate = (x_fp32 @ gate_weight.float()).reshape(intermediate_shape)
        up = (x_fp32 @ up_weight.float()).reshape(intermediate_shape)
        hidden = gate * torch.sigmoid(gate) * up
        output_2d = hidden.reshape(-1, intermediate_size) @ down_weight.float()
    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    output_shape = (*leading_shape, hidden_size)
    return Qwen3FFNStages(
        gate=gate,
        up=up,
        hidden=hidden,
        output=output_2d.reshape(output_shape),
    )


__all__ = [
    "QWEN3_8B_HIDDEN_SIZE",
    "QWEN3_8B_INTERMEDIATE_SIZE",
    "QWEN3_8B_TP2_INTERMEDIATE_SIZE",
    "Qwen3FFN",
    "Qwen3FFNProvenance",
    "Qwen3FFNStages",
    "build_qwen3_ffn",
    "qwen3_ffn_fp32_reference",
]
