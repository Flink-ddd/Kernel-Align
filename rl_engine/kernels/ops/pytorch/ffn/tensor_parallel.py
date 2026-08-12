# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tensor-parallel Qwen3-style SwiGLU FFN orchestration.

The module implements the non-sequence-parallel TP ownership contract used by
the Qwen3 dense MLP:

* ``gate`` and ``up`` are ColumnParallel (their output/intermediate dimension
  is sharded); their SwiGLU result stays local to the TP rank.
* ``down`` is RowParallel (its input/intermediate dimension is sharded); its
  same-coordinate hidden outputs are summed once over the explicit TP group.

The autograd collective mappings are deliberately asymmetric.  A RowParallel
forward reduction has an identity backward, while the replicated FFN input is
copied in the forward and reduced in the backward.  Consequently the sole
backward TP SUM occurs after the local Gate and Up input-gradient
contributions have been accumulated, exactly as required by the TP FFN
derivation.  There is no TP reduction for Down's feature-sharded ``dHidden``.

CP/SP configuration and CP weight-gradient reductions intentionally do not
belong in this PR3 module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn

from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSwiGLUOp

LocalGemm = Callable[[Tensor, Tensor], Tensor]


def _missing_deterministic_gemm(input_: Tensor, weight: Tensor) -> Tensor:
    """Fail closed rather than silently losing the batch-invariance contract."""

    del input_, weight
    raise RuntimeError(
        "TensorParallelFFN requires an explicit batch-invariant local GEMM. "
        "Pass the existing deterministic_gemm CUDA/Triton primitive with "
        "signature gemm(a[M, K], b[K, N]) -> [M, N]."
    )


@dataclass(frozen=True)
class FFNContext:
    """Explicit tensor-parallel configuration owned by an FFN caller.

    ``tp_group`` is never created or inferred by this module.  A multi-rank
    configuration must supply an initialized explicit process group; this is
    important because later PRs add distinct CP and SP groups.
    """

    tp_group: Any = None
    tp_size: Optional[int] = None
    tp_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.tp_group is None:
            size = 1 if self.tp_size is None else int(self.tp_size)
            rank = 0 if self.tp_rank is None else int(self.tp_rank)
            if size != 1 or rank != 0:
                raise ValueError(
                    "FFNContext(tp_group=None) only supports tp_size=1 and tp_rank=0. "
                    "Supply an explicit initialized tp_group for TP > 1."
                )
        else:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "FFNContext(tp_group=...) requires torch.distributed to be initialized."
                )
            group_size = dist.get_world_size(group=self.tp_group)
            group_rank = dist.get_rank(group=self.tp_group)
            size = group_size if self.tp_size is None else int(self.tp_size)
            rank = group_rank if self.tp_rank is None else int(self.tp_rank)
            if size != group_size:
                raise ValueError(
                    f"ctx.tp_size={size} does not match tp_group world size={group_size}."
                )
            if rank != group_rank:
                raise ValueError(f"ctx.tp_rank={rank} does not match tp_group rank={group_rank}.")

        if size < 1 or not 0 <= rank < size:
            raise ValueError(f"invalid TP coordinates: tp_size={size}, tp_rank={rank}.")
        object.__setattr__(self, "tp_size", size)
        object.__setattr__(self, "tp_rank", rank)

    @property
    def is_tensor_parallel(self) -> bool:
        """Whether this context owns more than one tensor-parallel shard."""

        assert self.tp_size is not None
        return self.tp_size > 1


def _all_reduce_sum(tensor: Tensor, ctx: FFNContext) -> Tensor:
    """Synchronously sum a tensor over the explicitly configured TP group."""

    if ctx.is_tensor_parallel:
        # TODO(WS2): replace NCCL/Gloo with the deterministic collective once it lands.
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=ctx.tp_group)
    return tensor


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Replicated input: identity forward, TP SUM backward.

    The Gate and Up ColumnParallel projections each produce a local
    same-coordinate ``dX`` contribution.  Autograd accumulates those local
    contributions before this function's backward executes, so this is one
    logical TP all-reduce of their combined ``[... , H]`` gradient.
    """

    @staticmethod
    def forward(ctx: Any, input_: Tensor, tp_ctx: FFNContext) -> Tensor:
        ctx.tp_ctx = tp_ctx
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        # Do not mutate a gradient owned by a downstream autograd node.
        grad_input = grad_output.contiguous().clone()
        return _all_reduce_sum(grad_input, ctx.tp_ctx), None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """RowParallel output: TP SUM forward, identity backward.

    Its backward must not reduce ``dOutput``.  Each rank receives the replicated
    top gradient and computes its local feature-sharded ``dHidden`` through
    Down's local weight shard.
    """

    @staticmethod
    def forward(ctx: Any, input_: Tensor, tp_ctx: FFNContext) -> Tensor:
        # The local Down partial is useful to callers through ``forward_local``;
        # preserve it by reducing a clone rather than modifying it in place.
        return _all_reduce_sum(input_.contiguous().clone(), tp_ctx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output, None


def _copy_to_tensor_parallel_region(input_: Tensor, ctx: FFNContext) -> Tensor:
    return _CopyToTensorParallelRegion.apply(input_, ctx)


def _reduce_from_tensor_parallel_region(input_: Tensor, ctx: FFNContext) -> Tensor:
    return _ReduceFromTensorParallelRegion.apply(input_, ctx)


def shard_qwen3_ffn_weights(
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    ctx: FFNContext,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return this rank's Qwen ``[out, in]`` Gate/Up/Down weight shards.

    Gate and Up are split along output rows.  Down is split along input columns.
    The returned tensors are contiguous views copied only as needed by callers
    that register them as local parameters.
    """

    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("gate_weight, up_weight, and down_weight must all be rank-2 tensors.")

    assert ctx.tp_size is not None
    assert ctx.tp_rank is not None
    intermediate_size, hidden_size = gate_weight.shape
    if up_weight.shape != (intermediate_size, hidden_size):
        raise ValueError(
            "up_weight must have the same [intermediate, hidden] shape as gate_weight; "
            f"got {tuple(up_weight.shape)} versus {tuple(gate_weight.shape)}."
        )
    if down_weight.shape != (hidden_size, intermediate_size):
        raise ValueError(
            "down_weight must have shape [hidden, intermediate]; "
            f"expected {(hidden_size, intermediate_size)}, got {tuple(down_weight.shape)}."
        )
    if intermediate_size % ctx.tp_size != 0:
        raise ValueError(
            f"intermediate_size={intermediate_size} must divide evenly by tp_size={ctx.tp_size}."
        )
    if not (gate_weight.device == up_weight.device == down_weight.device):
        raise ValueError("all FFN weights must be on the same device.")
    if not (gate_weight.dtype == up_weight.dtype == down_weight.dtype):
        raise ValueError("all FFN weights must have the same dtype.")

    local_intermediate = intermediate_size // ctx.tp_size
    start = ctx.tp_rank * local_intermediate
    stop = start + local_intermediate
    return (
        gate_weight[start:stop].contiguous(),
        up_weight[start:stop].contiguous(),
        down_weight[:, start:stop].contiguous(),
    )


class TensorParallelFFN(nn.Module):
    """Qwen3 SwiGLU FFN with ColumnParallel Gate/Up and RowParallel Down.

    Parameters use the normal ``torch.nn.Linear`` ``[out, in]`` layout.  This
    module owns one TP shard only; use :meth:`from_full_weights` in tests or a
    model loader to materialize rank-local parameter shards from full weights.

    ``gemm`` has the deterministic-GEMM-compatible signature
    ``gemm(a[M, K], b[K, N]) -> [M, N]``.  Production BF16 callers must inject
    the existing deterministic GEMM primitive (CUDA or Triton).  Omitting it
    fails closed when ``forward`` is called, because generic ``torch.matmul``
    would silently violate the batch-invariance contract.  This class
    intentionally does not introduce a separate GEMM arithmetic implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        ctx: Optional[FFNContext] = None,
        gate_weight: Optional[Tensor] = None,
        up_weight: Optional[Tensor] = None,
        down_weight: Optional[Tensor] = None,
        activation: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        gemm: Optional[LocalGemm] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or intermediate_size < 1:
            raise ValueError("hidden_size and intermediate_size must both be positive.")

        self.ctx = FFNContext() if ctx is None else ctx
        assert self.ctx.tp_size is not None
        if intermediate_size % self.ctx.tp_size != 0:
            raise ValueError(
                f"intermediate_size={intermediate_size} must divide evenly by "
                f"ctx.tp_size={self.ctx.tp_size}."
            )

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.local_intermediate_size = intermediate_size // self.ctx.tp_size
        self.activation = NativeSwiGLUOp() if activation is None else activation
        self._gemm: LocalGemm = _missing_deterministic_gemm if gemm is None else gemm

        self.gate_weight = self._make_parameter(
            gate_weight,
            (self.local_intermediate_size, hidden_size),
            "gate_weight",
            device=device,
            dtype=dtype,
        )
        self.up_weight = self._make_parameter(
            up_weight,
            (self.local_intermediate_size, hidden_size),
            "up_weight",
            device=device,
            dtype=dtype,
        )
        self.down_weight = self._make_parameter(
            down_weight,
            (hidden_size, self.local_intermediate_size),
            "down_weight",
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _make_parameter(
        value: Optional[Tensor],
        shape: tuple[int, int],
        name: str,
        *,
        device: Optional[torch.device | str],
        dtype: Optional[torch.dtype],
    ) -> nn.Parameter:
        if value is None:
            parameter = torch.empty(shape, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(parameter, a=5**0.5)
            return nn.Parameter(parameter)

        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
        if device is not None and value.device != torch.device(device):
            raise ValueError(
                f"{name} is on {value.device}, expected device {torch.device(device)}."
            )
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"{name} has dtype {value.dtype}, expected {dtype}.")
        return nn.Parameter(value.detach().clone())

    @classmethod
    def from_full_weights(
        cls,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        *,
        ctx: Optional[FFNContext] = None,
        activation: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        gemm: Optional[LocalGemm] = None,
    ) -> "TensorParallelFFN":
        """Construct a rank-local FFN module from full Qwen-format weights."""

        context = FFNContext() if ctx is None else ctx
        gate_shard, up_shard, down_shard = shard_qwen3_ffn_weights(
            gate_weight, up_weight, down_weight, ctx=context
        )
        intermediate_size, hidden_size = gate_weight.shape
        return cls(
            hidden_size,
            intermediate_size,
            ctx=context,
            gate_weight=gate_shard,
            up_weight=up_shard,
            down_weight=down_shard,
            activation=activation,
            gemm=gemm,
        )

    def _local_linear(self, input_: Tensor, weight: Tensor) -> Tensor:
        if input_.shape[-1] != weight.shape[1]:
            raise ValueError(
                f"input last dimension {input_.shape[-1]} does not match weight input "
                f"dimension {weight.shape[1]}."
            )
        input_2d = input_.reshape(-1, input_.shape[-1])
        output_2d = self._gemm(input_2d, weight.t().contiguous())
        expected_shape = (input_2d.shape[0], weight.shape[0])
        if output_2d.shape != expected_shape:
            raise RuntimeError(
                "gemm returned an invalid shape: expected "
                f"{expected_shape}, got {tuple(output_2d.shape)}."
            )
        return output_2d.reshape(*input_.shape[:-1], weight.shape[0])

    def forward_local(self, input_: Tensor) -> Tensor:
        """Return the local pre-TP-reduction Down output partial.

        This method is primarily a validation boundary: its result has shape
        ``[..., hidden_size]`` but represents only this rank's RowParallel
        contribution.  ``forward`` performs the required TP SUM over it.
        """

        if input_.ndim < 2:
            raise ValueError(
                f"input must have at least [tokens, hidden] dimensions, got {tuple(input_.shape)}."
            )
        if input_.device != self.gate_weight.device:
            raise ValueError(
                f"input is on {input_.device}, but FFN weights are on {self.gate_weight.device}."
            )
        if input_.dtype != self.gate_weight.dtype:
            raise ValueError(
                f"input has dtype {input_.dtype}, but FFN weights have dtype "
                f"{self.gate_weight.dtype}."
            )

        replicated_input = _copy_to_tensor_parallel_region(input_, self.ctx)
        gate = self._local_linear(replicated_input, self.gate_weight)
        up = self._local_linear(replicated_input, self.up_weight)
        hidden = self.activation(gate, up)
        return self._local_linear(hidden, self.down_weight)

    def forward(self, input_: Tensor) -> Tensor:
        """Compute the replicated hidden output after the one Down TP SUM."""

        local_output_partial = self.forward_local(input_)
        return _reduce_from_tensor_parallel_region(local_output_partial, self.ctx)
