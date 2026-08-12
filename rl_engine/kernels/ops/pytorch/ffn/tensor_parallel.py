# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tensor-, context-, and sequence-parallel Qwen3 SwiGLU FFN orchestration.

The module implements the non-sequence-parallel TP ownership contract used by
the Qwen3 dense MLP:

* ``gate`` and ``up`` are ColumnParallel (their output/intermediate dimension
  is sharded); their SwiGLU result stays local to the TP rank.
* ``down`` is RowParallel (its input/intermediate dimension is sharded); its
  same-coordinate hidden outputs are summed once over the explicit TP group.

Without SP, the autograd collective mappings are deliberately asymmetric: a
RowParallel forward reduction has an identity backward, while the replicated
FFN input is copied in the forward and reduced in the backward. Consequently
the sole backward TP SUM occurs after the local Gate and Up input-gradient
contributions have been accumulated. There is no TP reduction for Down's
feature-sharded ``dHidden``.

CP shards token rows. It has no forward activation collective: each CP rank
keeps its own rows. In backward, each TP-local parameter shard is replicated
across its CP lane, so Gate, Up, and Down each perform one CP SUM of ``dW``.

SP reuses the two ranks in a TP row but changes activation ownership. The
input/output residual stream is sequence-sharded; the local FFN body runs on
the SP AllGather result. Down's SP ReduceScatter(SUM) simultaneously restores
the sequence shard and sums the RowParallel output partials. Its autograd
mapping is AllGather, while the input AllGather's mapping is
ReduceScatter(SUM), so the gradient returned to RMSNorm remains SP-sharded.
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


def _resolve_parallel_coordinates(
    name: str,
    group: Any,
    size: Optional[int],
    rank: Optional[int],
) -> tuple[int, int]:
    """Validate one explicit parallel group and return concrete coordinates."""

    if group is None:
        resolved_size = 1 if size is None else int(size)
        resolved_rank = 0 if rank is None else int(rank)
        if resolved_size != 1 or resolved_rank != 0:
            raise ValueError(
                f"FFNContext({name}_group=None) only supports {name}_size=1 and "
                f"{name}_rank=0. Supply an explicit initialized {name}_group for "
                f"{name.upper()} > 1."
            )
    else:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                f"FFNContext({name}_group=...) requires torch.distributed to be initialized."
            )
        group_size = dist.get_world_size(group=group)
        group_rank = dist.get_rank(group=group)
        resolved_size = group_size if size is None else int(size)
        resolved_rank = group_rank if rank is None else int(rank)
        if resolved_size != group_size:
            raise ValueError(
                f"ctx.{name}_size={resolved_size} does not match {name}_group "
                f"world size={group_size}."
            )
        if resolved_rank != group_rank:
            raise ValueError(
                f"ctx.{name}_rank={resolved_rank} does not match {name}_group "
                f"rank={group_rank}."
            )

    if resolved_size < 1 or not 0 <= resolved_rank < resolved_size:
        raise ValueError(
            f"invalid {name.upper()} coordinates: {name}_size={resolved_size}, "
            f"{name}_rank={resolved_rank}."
        )
    return resolved_size, resolved_rank


@dataclass(frozen=True)
class FFNContext:
    """Explicit TP/CP/SP configuration owned by an FFN caller.

    Neither group is created or inferred by this module.  A multi-rank
    configuration must supply initialized explicit groups, which preserves the
    fixed TP/CP/SP topology.
    """

    tp_group: Any = None
    tp_size: Optional[int] = None
    tp_rank: Optional[int] = None
    cp_group: Any = None
    cp_size: Optional[int] = None
    cp_rank: Optional[int] = None
    sp_group: Any = None
    sp_size: Optional[int] = None
    sp_rank: Optional[int] = None

    def __post_init__(self) -> None:
        tp_size, tp_rank = _resolve_parallel_coordinates(
            "tp", self.tp_group, self.tp_size, self.tp_rank
        )
        cp_size, cp_rank = _resolve_parallel_coordinates(
            "cp", self.cp_group, self.cp_size, self.cp_rank
        )
        sp_size, sp_rank = _resolve_parallel_coordinates(
            "sp", self.sp_group, self.sp_size, self.sp_rank
        )
        if sp_size > 1 and (sp_size != tp_size or sp_rank != tp_rank):
            raise ValueError(
                "SP must align with the TP rank dimension: sp_size/sp_rank must match "
                "tp_size/tp_rank when sequence parallelism is enabled."
            )
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "tp_rank", tp_rank)
        object.__setattr__(self, "cp_size", cp_size)
        object.__setattr__(self, "cp_rank", cp_rank)
        object.__setattr__(self, "sp_size", sp_size)
        object.__setattr__(self, "sp_rank", sp_rank)

    @property
    def is_tensor_parallel(self) -> bool:
        """Whether this context owns more than one tensor-parallel shard."""

        assert self.tp_size is not None
        return self.tp_size > 1

    @property
    def is_context_parallel(self) -> bool:
        """Whether this context owns more than one context-parallel token shard."""

        assert self.cp_size is not None
        return self.cp_size > 1

    @property
    def is_sequence_parallel(self) -> bool:
        """Whether residual-stream activations are split across the SP group."""

        assert self.sp_size is not None
        return self.sp_size > 1


def _all_reduce_sum(tensor: Tensor, group: Any, world_size: int) -> Tensor:
    """Synchronously sum a tensor over an explicitly configured process group."""

    if world_size > 1:
        # TODO(WS2): replace NCCL/Gloo with the deterministic collective once it lands.
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def _all_reduce_tp_sum(tensor: Tensor, ctx: FFNContext) -> Tensor:
    assert ctx.tp_size is not None
    return _all_reduce_sum(tensor, ctx.tp_group, ctx.tp_size)


def _all_reduce_cp_sum(tensor: Tensor, ctx: FFNContext) -> Tensor:
    assert ctx.cp_size is not None
    return _all_reduce_sum(tensor, ctx.cp_group, ctx.cp_size)


def _all_reduce_sp_sum(tensor: Tensor, ctx: FFNContext) -> Tensor:
    assert ctx.sp_size is not None
    return _all_reduce_sum(tensor, ctx.sp_group, ctx.sp_size)


def _all_gather_sequence(tensor: Tensor, ctx: FFNContext) -> Tensor:
    """Gather equal sequence shards along the penultimate tensor dimension."""

    if not ctx.is_sequence_parallel:
        return tensor
    if tensor.ndim < 2:
        raise ValueError(
            f"SP AllGather requires [..., sequence, hidden], got {tuple(tensor.shape)}."
        )

    assert ctx.sp_size is not None
    gathered = [torch.empty_like(tensor) for _ in range(ctx.sp_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=ctx.sp_group)
    return torch.cat(gathered, dim=-2)


def _reduce_scatter_sequence_sum(tensor: Tensor, ctx: FFNContext) -> Tensor:
    """SUM-reduce equal full-sequence tensors then return this rank's sequence shard.

    NCCL uses its native ``reduce_scatter`` collective. Gloo deployments that
    do not implement that primitive use an equivalent all-reduce plus local
    sequence slice solely as a portable test fallback.
    """

    if not ctx.is_sequence_parallel:
        return tensor
    if tensor.ndim < 2:
        raise ValueError(
            f"SP ReduceScatter requires [..., sequence, hidden], got {tuple(tensor.shape)}."
        )

    assert ctx.sp_size is not None
    assert ctx.sp_rank is not None
    sequence_size = tensor.shape[-2]
    if sequence_size % ctx.sp_size != 0:
        raise ValueError(
            f"sequence size={sequence_size} must divide evenly by sp_size={ctx.sp_size}."
        )

    chunks = [chunk.contiguous() for chunk in tensor.chunk(ctx.sp_size, dim=-2)]
    output = torch.empty_like(chunks[0])
    try:
        dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM, group=ctx.sp_group)
        return output
    except RuntimeError as error:
        # ProcessGroupGloo lacks reduce_scatter in some supported PyTorch builds.
        if "does not support reduce_scatter" not in str(error).lower():
            raise
        reduced = _all_reduce_sp_sum(tensor.contiguous(), ctx)
        local_sequence = sequence_size // ctx.sp_size
        return reduced.narrow(-2, ctx.sp_rank * local_sequence, local_sequence).contiguous()


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Replicated input: identity forward, TP SUM backward when SP is disabled.

    The Gate and Up ColumnParallel projections each produce a local
    same-coordinate ``dX`` contribution.  Autograd accumulates those local
    contributions before this function's backward executes, so this is one
    logical TP all-reduce of their combined ``[... , H]`` gradient. With SP,
    the preceding AllGather's backward owns that SUM via ReduceScatter instead.
    """

    @staticmethod
    def forward(ctx: Any, input_: Tensor, tp_ctx: FFNContext) -> Tensor:
        ctx.tp_ctx = tp_ctx
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        # Do not mutate a gradient owned by a downstream autograd node.
        grad_input = grad_output.contiguous().clone()
        if ctx.tp_ctx.is_sequence_parallel:
            # The preceding SP AllGather owns the single necessary dX SUM and
            # sequence split in its backward ReduceScatter.
            return grad_input, None
        return _all_reduce_tp_sum(grad_input, ctx.tp_ctx), None


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
        return _all_reduce_tp_sum(input_.contiguous().clone(), tp_ctx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output, None


def _copy_to_tensor_parallel_region(input_: Tensor, ctx: FFNContext) -> Tensor:
    return _CopyToTensorParallelRegion.apply(input_, ctx)


def _reduce_from_tensor_parallel_region(input_: Tensor, ctx: FFNContext) -> Tensor:
    return _ReduceFromTensorParallelRegion.apply(input_, ctx)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """SP AllGather forward; SUM ReduceScatter backward."""

    @staticmethod
    def forward(ctx: Any, input_: Tensor, sp_ctx: FFNContext) -> Tensor:
        ctx.sp_ctx = sp_ctx
        return _all_gather_sequence(input_, sp_ctx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return _reduce_scatter_sequence_sum(grad_output, ctx.sp_ctx), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """SP SUM ReduceScatter forward; AllGather backward."""

    @staticmethod
    def forward(ctx: Any, input_: Tensor, sp_ctx: FFNContext) -> Tensor:
        ctx.sp_ctx = sp_ctx
        return _reduce_scatter_sequence_sum(input_, sp_ctx)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return _all_gather_sequence(grad_output, ctx.sp_ctx), None


def _gather_from_sequence_parallel_region(input_: Tensor, ctx: FFNContext) -> Tensor:
    return _GatherFromSequenceParallelRegion.apply(input_, ctx)


def _reduce_scatter_to_sequence_parallel_region(input_: Tensor, ctx: FFNContext) -> Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(input_, ctx)


class _CopyWeightToContextParallelRegion(torch.autograd.Function):
    """Replicated parameter: identity forward, CP SUM of its local ``dW`` backward.

    CP shards token rows, not parameter coordinates.  Each rank therefore
    computes a local contribution to the same TP-local weight shard.  This
    function is attached individually to Down, Gate, and Up weights so their
    three logical gradient tensors each reduce over the explicit same-TP-lane
    CP group.
    """

    @staticmethod
    def forward(ctx: Any, weight: Tensor, cp_ctx: FFNContext) -> Tensor:
        ctx.cp_ctx = cp_ctx
        return weight

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        grad_weight = grad_output.contiguous().clone()
        return _all_reduce_cp_sum(grad_weight, ctx.cp_ctx), None


def _copy_weight_to_context_parallel_region(weight: Tensor, ctx: FFNContext) -> Tensor:
    return _CopyWeightToContextParallelRegion.apply(weight, ctx)


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
    """Qwen3 SwiGLU FFN with TP feature shards and CP-local token rows.

    Parameters use the normal ``torch.nn.Linear`` ``[out, in]`` layout.  This
    module owns one TP shard; that shard is replicated across CP ranks. The
    caller provides CP-local token rows and can use :meth:`from_full_weights`
    in tests or a model loader to materialize rank-local parameter shards.

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
        """Return the local pre-TP/SP-reduction Down output partial.

        This method is primarily a validation boundary: its result has shape
        ``[..., hidden_size]`` but represents only this rank's RowParallel
        contribution. ``forward`` performs the required TP SUM or SP
        ReduceScatter(SUM) ownership conversion over it.
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

        gathered_input = _gather_from_sequence_parallel_region(input_, self.ctx)
        replicated_input = _copy_to_tensor_parallel_region(gathered_input, self.ctx)
        gate_weight = _copy_weight_to_context_parallel_region(self.gate_weight, self.ctx)
        up_weight = _copy_weight_to_context_parallel_region(self.up_weight, self.ctx)
        down_weight = _copy_weight_to_context_parallel_region(self.down_weight, self.ctx)
        gate = self._local_linear(replicated_input, gate_weight)
        up = self._local_linear(replicated_input, up_weight)
        hidden = self.activation(gate, up)
        return self._local_linear(hidden, down_weight)

    def forward(self, input_: Tensor) -> Tensor:
        """Compute the residual-stream output in the configured TP/SP ownership."""

        local_output_partial = self.forward_local(input_)
        if self.ctx.is_sequence_parallel:
            # SP RS(SUM) is the TP RowParallel output reduction and restores
            # the residual stream's local sequence shard in one collective.
            return _reduce_scatter_to_sequence_parallel_region(local_output_partial, self.ctx)
        return _reduce_from_tensor_parallel_region(local_output_partial, self.ctx)
