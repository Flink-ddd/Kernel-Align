# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic context-parallel attention reference.

This module is the correctness-first WS2 reference for CP-aware standard
softmax attention. It intentionally stays in PyTorch and uses fp32 partial
states so fused CUDA/Triton backends can validate their CP/LSE merge semantics
against a small, inspectable implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from rl_engine.kernels.attention_contract import (
    SplitKVExecutionPlan,
    SplitKVMode,
    SplitKVRuntimeCoordinate,
    SplitKVRuntimePlanEntry,
    SplitKVRuntimePlanSet,
)
from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp


@dataclass(frozen=True)
class AttentionPartialState:
    """One KV block's attention state before deterministic LSE merge.

    ``out`` is already normalized within the local KV block and has shape
    ``[B, Hq, Sq, D]``. ``lse`` is the local attention-domain log-sum-exp with
    shape ``[B, Hq, Sq]``. ``block_start`` / ``block_end`` are logical global KV
    positions and define the canonical merge order.
    """

    out: torch.Tensor
    lse: torch.Tensor
    block_start: int
    block_end: int

    def __post_init__(self) -> None:
        if self.out.ndim != 4:
            raise ValueError("partial attention out must have shape [B, Hq, Sq, D]")
        if self.lse.shape != self.out.shape[:3]:
            raise ValueError("partial attention lse must have shape [B, Hq, Sq]")
        if self.out.device != self.lse.device:
            raise ValueError("partial attention out/lse must be on the same device")
        if self.out.dtype is not torch.float32 or self.lse.dtype is not torch.float32:
            raise ValueError("partial attention out/lse must remain FP32 before merge")
        if self.block_start < 0:
            raise ValueError("block_start must be non-negative")
        if self.block_end < self.block_start:
            raise ValueError("block_end must be >= block_start")


@dataclass(frozen=True)
class AttentionBackwardGradients:
    """Training-side gradients emitted by the CP attention backward reference."""

    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


@dataclass(frozen=True)
class AttentionBackwardPathResult:
    """One materialized CP attention backward path."""

    name: str
    out: torch.Tensor
    lse: torch.Tensor
    gradients: AttentionBackwardGradients
    provenance: dict[str, object]


@dataclass(frozen=True)
class GradientDriftStats:
    """Shape-aware absolute drift summary for backward validation reports."""

    max_abs: float
    mean_abs: float
    p95_abs: float
    p99_abs: float
    active_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "p95_abs": self.p95_abs,
            "p99_abs": self.p99_abs,
            "active_count": self.active_count,
        }


@dataclass(frozen=True)
class AttentionBackwardRankDrift:
    """Backward drift for one logical CP rank's sequence ownership."""

    rank: int
    dq: GradientDriftStats
    dk: GradientDriftStats
    dv: GradientDriftStats

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "dq": self.dq.to_dict(),
            "dk": self.dk.to_dict(),
            "dv": self.dv.to_dict(),
        }


@dataclass(frozen=True)
class AttentionBackwardPathDrift:
    """Candidate-vs-reference backward drift for one CP path."""

    candidate_name: str
    dq: GradientDriftStats
    dk: GradientDriftStats
    dv: GradientDriftStats
    out: GradientDriftStats
    lse: GradientDriftStats
    per_rank: tuple[AttentionBackwardRankDrift, ...]
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "dq": self.dq.to_dict(),
            "dk": self.dk.to_dict(),
            "dv": self.dv.to_dict(),
            "out": self.out.to_dict(),
            "lse": self.lse.to_dict(),
            "per_rank": [item.to_dict() for item in self.per_rank],
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class AttentionBackwardComparisonReport:
    """Structured PR8 report for CP attention gradient drift validation."""

    reference_name: str
    drifts: tuple[AttentionBackwardPathDrift, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_name": self.reference_name,
            "drifts": [drift.to_dict() for drift in self.drifts],
        }


def merge_attention_partial_states(
    states: Sequence[AttentionPartialState],
) -> AttentionPartialState:
    """Merge CP/chunk partial states in logical block order.

    The merge is the online-softmax/LSE merge used by attention, not a plain
    sum. The input order is deliberately ignored: states are sorted by logical
    ``block_start`` so the result depends on global block indices rather than
    arrival order.
    """

    if not states:
        raise ValueError("at least one attention partial state is required")

    ordered = sorted(states, key=lambda item: (item.block_start, item.block_end))
    _validate_merge_shapes_and_ranges(ordered)

    merged = ordered[0]
    merged_out = merged.out.float()
    merged_lse = merged.lse.float()
    for state in ordered[1:]:
        merged_out, merged_lse = _merge_two_states(
            merged_out,
            merged_lse,
            state.out.float(),
            state.lse.float(),
        )

    return AttentionPartialState(
        out=merged_out,
        lse=merged_lse,
        block_start=ordered[0].block_start,
        block_end=ordered[-1].block_end,
    )


class DeterministicCPAttentionReferenceOp:
    """Correctness-first CP attention reference for prefill and chunked prefill.

    The reference consumes attention-ready Q/K. For Qwen3 WS2 this means Q/K
    have already passed QK-Norm and RoPE unless an outer contract explicitly
    marks them as pre-RoPE. RoPE is intentionally kept outside this CP merge
    implementation so fused and unfused ``RoPE+Attention`` paths can compare the
    same post-RoPE Q/K boundary before validating CP communication.

    The op emulates CP by splitting query and KV sequence dimensions into
    logical CP shards. Each query shard computes one partial attention state per
    KV block, then merges those states in fixed global-block order using fp32
    LSE arithmetic. ``forward`` returns the input dtype after the final write;
    ``forward_fp32`` keeps the fp32 merged output.
    """

    op_class = "attention"

    def __init__(self, *, strict_bitwise: bool = False) -> None:
        if not isinstance(strict_bitwise, bool):
            raise TypeError("strict_bitwise must be a bool")
        self.strict_bitwise = strict_bitwise

    @staticmethod
    def split_kv_execution_plans(
        total_kv_tokens: int,
        *,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> list[dict[str, object]]:
        """Export the actual logical Split-KV plan before execution."""

        return split_kv_execution_plan_provenance(
            total_kv_tokens,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            backend="deterministic_cp_reference",
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute CP attention with fp32 accumulation and final input-dtype write."""

        out, _ = self.forward_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            output_dtype=q.dtype,
        )
        return out

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute CP attention with fp32 accumulation and fp32 output."""

        out, _ = self.forward_fp32_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        )
        return out

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(out, lse)`` for the CP reference path.

        ``lse`` is always fp32 and in the attention domain. ``out`` is fp32
        until the final write, then downcast to ``output_dtype``. When omitted,
        ``output_dtype`` defaults to the input dtype.
        """

        resolved_output_dtype = q.dtype if output_dtype is None else output_dtype
        _validate_output_dtype(resolved_output_dtype)
        if self.strict_bitwise:
            out, lse = self._forward_strict_bitwise(
                q,
                k,
                v,
                causal=causal,
                scale=scale,
                key_padding_mask=key_padding_mask,
                query_position_offsets=query_position_offsets,
                key_position_offsets=key_position_offsets,
                kv_chunk_size=kv_chunk_size,
            )
        else:
            out, lse = self._forward_impl(
                q,
                k,
                v,
                causal=causal,
                scale=scale,
                key_padding_mask=key_padding_mask,
                query_position_offsets=query_position_offsets,
                key_position_offsets=key_position_offsets,
                cp_world_size=cp_world_size,
                kv_chunk_size=kv_chunk_size,
            )
        out = out.to(resolved_output_dtype)
        return out, lse

    def forward_fp32_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fp32 ``(out, lse)`` for the CP reference path."""

        return self.forward_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            output_dtype=torch.float32,
        )

    def _forward_strict_bitwise(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: Optional[float],
        key_padding_mask: Optional[torch.Tensor],
        query_position_offsets: Optional[torch.Tensor],
        key_position_offsets: Optional[torch.Tensor],
        kv_chunk_size: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one batch/CP-independent arithmetic schedule."""

        _validate_qkv(q, k, v)
        _validate_scale(scale)
        batch, hq, sq, dim = q.shape
        skv = k.size(2)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, skv):
                raise ValueError("key_padding_mask must have shape [B, Skv]")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")
        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            batch,
            q.device,
            default=skv - sq,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            batch,
            q.device,
            default=0,
            name="key_position_offsets",
        )
        kv_bounds = _kv_block_bounds(skv, 1, kv_chunk_size)
        out_rows: list[torch.Tensor] = []
        lse_rows: list[torch.Tensor] = []
        for batch_index in range(batch):
            q_batch = q[batch_index : batch_index + 1].contiguous()
            k_batch = k[batch_index : batch_index + 1].contiguous()
            v_batch = v[batch_index : batch_index + 1].contiguous()
            pad_batch = (
                None
                if key_padding_mask is None
                else key_padding_mask[batch_index : batch_index + 1].contiguous()
            )
            query_offset = query_offsets[batch_index : batch_index + 1]
            key_offset = key_offsets[batch_index : batch_index + 1]
            query_rows: list[torch.Tensor] = []
            lse_query_rows: list[torch.Tensor] = []
            for query_index in range(sq):
                q_row = q_batch[:, :, query_index : query_index + 1, :].contiguous()
                states = [
                    self.local_partial_state(
                        q_row,
                        k_batch[:, :, key_start:key_end, :].contiguous(),
                        v_batch[:, :, key_start:key_end, :].contiguous(),
                        q_start=query_index,
                        k_start=key_start,
                        total_kv_len=skv,
                        total_query_len=sq,
                        causal=causal,
                        scale=scale,
                        key_padding_mask=(
                            None
                            if pad_batch is None
                            else pad_batch[:, key_start:key_end].contiguous()
                        ),
                        query_position_offsets=query_offset,
                        key_position_offsets=key_offset,
                    )
                    for key_start, key_end in kv_bounds
                    if key_start != key_end
                ]
                merged = merge_attention_partial_states(states)
                query_rows.append(merged.out)
                lse_query_rows.append(merged.lse)
            out_rows.append(torch.cat(query_rows, dim=2))
            lse_rows.append(torch.cat(lse_query_rows, dim=2))
        return torch.cat(out_rows, dim=0), torch.cat(lse_rows, dim=0)

    def backward_reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
        cp_world_size: int = 1,
        kv_chunk_size: Optional[int] = None,
        output_dtype: Optional[torch.dtype] = torch.float32,
        name: Optional[str] = None,
    ) -> AttentionBackwardPathResult:
        """Run the deterministic training-side backward validation path.

        The semantic backward input is ``dout`` plus the forward attention state
        produced from the same Q/K/V, masks, position offsets, CP world, and KV
        block order. The reference keeps the softmax/merge math in fp32 and
        records the final-write dtype in provenance; decode backward is
        intentionally out of scope for PR8.
        """

        _validate_qkv(q, k, v)
        if dout.shape != q.shape:
            raise ValueError("dout must have shape [B, Hq, Sq, D], matching q")
        if not torch.is_floating_point(dout) or torch.is_complex(dout):
            raise ValueError("dout must be a real floating-point tensor")
        if dout.device != q.device:
            raise ValueError("dout must be on the same device as q, k, and v")
        if dout.dtype != q.dtype:
            raise ValueError("dout must have the same dtype as q")
        q_leaf = q.detach().clone().requires_grad_(True)
        k_leaf = k.detach().clone().requires_grad_(True)
        v_leaf = v.detach().clone().requires_grad_(True)

        resolved_output_dtype = q.dtype if output_dtype is None else output_dtype
        _validate_output_dtype(resolved_output_dtype)
        out, lse = self.forward_with_lse(
            q_leaf,
            k_leaf,
            v_leaf,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_offsets=query_position_offsets,
            key_position_offsets=key_position_offsets,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            output_dtype=resolved_output_dtype,
        )
        torch.autograd.backward(out, dout.to(dtype=out.dtype))
        if q_leaf.grad is None or k_leaf.grad is None or v_leaf.grad is None:
            raise RuntimeError("CP attention backward did not produce dq/dk/dv")

        return AttentionBackwardPathResult(
            name=name
            or _backward_path_name(cp_world_size=cp_world_size, kv_chunk_size=kv_chunk_size),
            out=out.detach(),
            lse=lse.detach(),
            gradients=AttentionBackwardGradients(
                dq=q_leaf.grad.detach(),
                dk=k_leaf.grad.detach(),
                dv=v_leaf.grad.detach(),
            ),
            provenance={
                "attention_mode": "prefill" if kv_chunk_size is None else "chunked_prefill",
                "gradient_mode": "training_backward",
                "gradient_inputs": ["q", "k", "v"],
                "gradient_outputs": ["out"],
                "saved_forward_state": [
                    "out",
                    "attention_lse",
                    "causal_mask",
                    "key_padding_mask",
                    "query_position_offsets",
                    "key_position_offsets",
                    "global_block_index",
                ],
                "cp_world_size": cp_world_size,
                "kv_chunk_size": kv_chunk_size,
                "requested_split_kv_policy": ("disabled" if kv_chunk_size is None else "fixed"),
                "requested_split_kv_size": kv_chunk_size,
                "actual_split_kv_plans": split_kv_execution_plan_provenance(
                    k.size(2),
                    cp_world_size=cp_world_size,
                    kv_chunk_size=kv_chunk_size,
                    backend="deterministic_cp_backward_reference",
                ),
                "merge_order": "global_block_index",
                "accum_dtype": "fp32",
                "downcast_at": "final_write",
                "output_dtype": str(resolved_output_dtype).replace("torch.", ""),
                "q_dtype": str(q.dtype).replace("torch.", ""),
                "k_dtype": str(k.dtype).replace("torch.", ""),
                "v_dtype": str(v.dtype).replace("torch.", ""),
                "dout_dtype": str(dout.dtype).replace("torch.", ""),
                "te_backward_oracle": "not_used",
                "decode_backward": "not_supported",
            },
        )

    def local_partial_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        q_start: int,
        k_start: int,
        total_kv_len: int,
        total_query_len: Optional[int] = None,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_position_offsets: Optional[torch.Tensor] = None,
        key_position_offsets: Optional[torch.Tensor] = None,
    ) -> AttentionPartialState:
        """Compute one query shard against one logical KV block.

        ``query_position_offsets`` and ``key_position_offsets`` are optional
        per-batch-row base positions. They let the reference express varlen or
        packed metadata while retaining the dense [B, H, S, D] tensor layout.
        For post-RoPE Q/K, these offsets must describe the same absolute token
        positions used when RoPE was applied.
        """

        _validate_qkv(q, k, v)
        _validate_scale(scale)
        if q_start < 0 or k_start < 0:
            raise ValueError("q_start and k_start must be non-negative")
        if total_kv_len < k_start + k.size(2):
            raise ValueError("total_kv_len must cover the local KV block")
        if total_query_len is None:
            total_query_len = q.size(2)
        if total_query_len < q_start + q.size(2):
            raise ValueError("total_query_len must cover the local query block")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (q.size(0), k.size(2)):
                raise ValueError("local key_padding_mask must have shape [B, local_skv]")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("local key_padding_mask must be bool")
        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            q.size(0),
            q.device,
            default=total_kv_len - total_query_len,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            q.size(0),
            q.device,
            default=0,
            name="key_position_offsets",
        )

        ctx = NativeAttentionOp._strict_fp32_math(q.device.type)
        with ctx:
            qf = q.float()
            kf = k.float()
            vf = v.float()
            hq, sq, dim = qf.shape[1], qf.shape[2], qf.shape[3]
            hkv, skv = kf.shape[1], kf.shape[2]
            if hkv != hq:
                repeat = hq // hkv
                kf = kf.repeat_interleave(repeat, dim=1)
                vf = vf.repeat_interleave(repeat, dim=1)

            if skv == 0:
                zero_dep = _zero_dependency(qf, kf, vf)
                return AttentionPartialState(
                    out=torch.zeros(q.size(0), hq, sq, dim, device=q.device, dtype=torch.float32)
                    + zero_dep,
                    lse=torch.full(
                        (q.size(0), hq, sq),
                        float("-inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    + zero_dep,
                    block_start=k_start,
                    block_end=k_start,
                )

            scale_value = scale if scale is not None else (1.0 / math.sqrt(dim))
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale_value
            if causal:
                query_base = query_offsets[:, None] + q_start
                key_base = key_offsets[:, None] + k_start
                q_pos = torch.arange(sq, device=q.device, dtype=torch.long) + query_base
                k_pos = torch.arange(skv, device=q.device, dtype=torch.long) + key_base
                causal_mask = k_pos[:, None, :] > q_pos[:, :, None]
                scores = scores.masked_fill(causal_mask[:, None, :, :], float("-inf"))
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

            lse = torch.logsumexp(scores, dim=-1)
            finite_lse = torch.isfinite(lse)
            weights = torch.exp(scores - lse.unsqueeze(-1))
            weights = torch.where(finite_lse.unsqueeze(-1), weights, torch.zeros_like(weights))
            out = torch.matmul(weights, vf)
            return AttentionPartialState(
                out=out,
                lse=lse,
                block_start=k_start,
                block_end=k_start + skv,
            )

    def _forward_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: Optional[float],
        key_padding_mask: Optional[torch.Tensor],
        query_position_offsets: Optional[torch.Tensor],
        key_position_offsets: Optional[torch.Tensor],
        cp_world_size: int,
        kv_chunk_size: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_qkv(q, k, v)
        _validate_scale(scale)
        if (
            isinstance(cp_world_size, bool)
            or not isinstance(cp_world_size, int)
            or cp_world_size < 1
        ):
            raise ValueError("cp_world_size must be >= 1")
        if kv_chunk_size is not None and (
            isinstance(kv_chunk_size, bool)
            or not isinstance(kv_chunk_size, int)
            or kv_chunk_size < 1
        ):
            raise ValueError("kv_chunk_size must be >= 1 when provided")

        batch, hq, sq, dim = q.shape
        skv = k.size(2)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, skv):
                raise ValueError("key_padding_mask must have shape [B, Skv]")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool")
        query_offsets = _normalize_position_offsets(
            query_position_offsets,
            batch,
            q.device,
            default=skv - sq,
            name="query_position_offsets",
        )
        key_offsets = _normalize_position_offsets(
            key_position_offsets,
            batch,
            q.device,
            default=0,
            name="key_position_offsets",
        )

        q_bounds = _split_bounds(sq, cp_world_size)
        kv_bounds = _kv_block_bounds(skv, cp_world_size, kv_chunk_size)
        out_chunks: list[torch.Tensor] = []
        lse_chunks: list[torch.Tensor] = []
        for q_start, q_end in q_bounds:
            if q_start == q_end:
                continue
            q_block = q[:, :, q_start:q_end, :]
            states = [
                self.local_partial_state(
                    q_block,
                    k[:, :, k_start:k_end, :],
                    v[:, :, k_start:k_end, :],
                    q_start=q_start,
                    k_start=k_start,
                    total_kv_len=skv,
                    total_query_len=sq,
                    causal=causal,
                    scale=scale,
                    key_padding_mask=(
                        None if key_padding_mask is None else key_padding_mask[:, k_start:k_end]
                    ),
                    query_position_offsets=query_offsets,
                    key_position_offsets=key_offsets,
                )
                for k_start, k_end in kv_bounds
                if k_start != k_end
            ]
            if states:
                merged = merge_attention_partial_states(states)
                out_chunks.append(merged.out)
                lse_chunks.append(merged.lse)
            else:
                zero_dep = _zero_dependency(q_block.float(), k.float(), v.float())
                out_chunks.append(
                    torch.zeros(batch, hq, q_end - q_start, dim, device=q.device) + zero_dep
                )
                lse_chunks.append(
                    torch.full(
                        (batch, hq, q_end - q_start),
                        float("-inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    + zero_dep
                )

        if not out_chunks:
            zero_dep = _zero_dependency(q.float(), k.float(), v.float())
            return (
                torch.empty(batch, hq, 0, dim, device=q.device, dtype=torch.float32) + zero_dep,
                torch.empty(batch, hq, 0, device=q.device, dtype=torch.float32) + zero_dep,
            )
        return torch.cat(out_chunks, dim=2), torch.cat(lse_chunks, dim=2)


def compare_cp_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    query_position_offsets: Optional[torch.Tensor] = None,
    key_position_offsets: Optional[torch.Tensor] = None,
    candidate_cp_world_size: int = 2,
    candidate_kv_chunk_size: Optional[int] = None,
    output_dtype: Optional[torch.dtype] = torch.float32,
) -> AttentionBackwardComparisonReport:
    """Compare CP=1 backward with a CP/chunked-prefill candidate.

    The report includes whole-tensor ``dq/dk/dv`` drift and per-logical-CP-rank
    slices. It is a validation/reporting helper, not a separate production
    backward kernel.
    """

    op = DeterministicCPAttentionReferenceOp()
    reference = op.backward_reference(
        q,
        k,
        v,
        dout,
        causal=causal,
        scale=scale,
        key_padding_mask=key_padding_mask,
        query_position_offsets=query_position_offsets,
        key_position_offsets=key_position_offsets,
        cp_world_size=1,
        kv_chunk_size=None,
        output_dtype=output_dtype,
        name="cp1_backward_reference",
    )
    candidate = op.backward_reference(
        q,
        k,
        v,
        dout,
        causal=causal,
        scale=scale,
        key_padding_mask=key_padding_mask,
        query_position_offsets=query_position_offsets,
        key_position_offsets=key_position_offsets,
        cp_world_size=candidate_cp_world_size,
        kv_chunk_size=candidate_kv_chunk_size,
        output_dtype=output_dtype,
    )
    return AttentionBackwardComparisonReport(
        reference_name=reference.name,
        drifts=(_compare_backward_path(candidate, reference),),
    )


def _compare_backward_path(
    candidate: AttentionBackwardPathResult,
    reference: AttentionBackwardPathResult,
) -> AttentionBackwardPathDrift:
    cp_world_size = _provenance_int(candidate.provenance, "cp_world_size")
    return AttentionBackwardPathDrift(
        candidate_name=candidate.name,
        dq=_drift_stats(candidate.gradients.dq, reference.gradients.dq),
        dk=_drift_stats(candidate.gradients.dk, reference.gradients.dk),
        dv=_drift_stats(candidate.gradients.dv, reference.gradients.dv),
        out=_drift_stats(candidate.out, reference.out),
        lse=_drift_stats(candidate.lse, reference.lse),
        per_rank=_per_rank_backward_drifts(candidate, reference, cp_world_size),
        provenance=candidate.provenance,
    )


def _per_rank_backward_drifts(
    candidate: AttentionBackwardPathResult,
    reference: AttentionBackwardPathResult,
    cp_world_size: int,
) -> tuple[AttentionBackwardRankDrift, ...]:
    q_bounds = _split_bounds(candidate.gradients.dq.size(2), cp_world_size)
    kv_bounds = _split_bounds(candidate.gradients.dk.size(2), cp_world_size)
    per_rank = []
    for rank, ((q_start, q_end), (kv_start, kv_end)) in enumerate(zip(q_bounds, kv_bounds)):
        per_rank.append(
            AttentionBackwardRankDrift(
                rank=rank,
                dq=_drift_stats(
                    candidate.gradients.dq[:, :, q_start:q_end, :],
                    reference.gradients.dq[:, :, q_start:q_end, :],
                ),
                dk=_drift_stats(
                    candidate.gradients.dk[:, :, kv_start:kv_end, :],
                    reference.gradients.dk[:, :, kv_start:kv_end, :],
                ),
                dv=_drift_stats(
                    candidate.gradients.dv[:, :, kv_start:kv_end, :],
                    reference.gradients.dv[:, :, kv_start:kv_end, :],
                ),
            )
        )
    return tuple(per_rank)


def _drift_stats(candidate: torch.Tensor, reference: torch.Tensor) -> GradientDriftStats:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match "
            f"reference shape {tuple(reference.shape)}"
        )
    diff = (candidate.float() - reference.float()).abs().reshape(-1)
    active_count = int(diff.numel())
    if active_count == 0:
        return GradientDriftStats(0.0, 0.0, 0.0, 0.0, 0)
    return GradientDriftStats(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        p95_abs=float(torch.quantile(diff, 0.95).item()),
        p99_abs=float(torch.quantile(diff, 0.99).item()),
        active_count=active_count,
    )


def _backward_path_name(*, cp_world_size: int, kv_chunk_size: Optional[int]) -> str:
    prefix = f"cp{cp_world_size}"
    if kv_chunk_size is None:
        return f"{prefix}_backward"
    return f"{prefix}_chunked_backward"


def _provenance_int(provenance: dict[str, object], key: str) -> int:
    value = provenance[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"provenance field {key!r} must be an int")
    return value


def _merge_two_states(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_lse = torch.logaddexp(lse_a, lse_b)
    finite = torch.isfinite(merged_lse)
    weight_a = torch.where(finite, torch.exp(lse_a - merged_lse), torch.zeros_like(merged_lse))
    weight_b = torch.where(finite, torch.exp(lse_b - merged_lse), torch.zeros_like(merged_lse))
    merged_out = weight_a.unsqueeze(-1) * out_a + weight_b.unsqueeze(-1) * out_b
    return merged_out, merged_lse


def _validate_merge_shapes_and_ranges(states: Sequence[AttentionPartialState]) -> None:
    first = states[0]
    previous_end = first.block_end
    for state in states[1:]:
        if state.out.shape != first.out.shape or state.lse.shape != first.lse.shape:
            raise ValueError("all partial states must have matching out/lse shapes")
        if state.block_start != previous_end:
            raise ValueError("partial state block ranges must be gap-free and non-overlapping")
        previous_end = state.block_end


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, S, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.size(0) != k.size(0) or q.size(3) != k.size(3):
        raise ValueError("q, k, and v must share batch size and head dim")
    if q.size(1) < 1 or k.size(1) < 1 or q.size(3) < 1:
        raise ValueError("q, k, and v must have positive head counts and head dim")
    if not all(torch.is_floating_point(tensor) for tensor in (q, k, v)) or any(
        torch.is_complex(tensor) for tensor in (q, k, v)
    ):
        raise ValueError("q, k, and v must be real floating-point tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.size(1) % k.size(1) != 0:
        raise ValueError(f"Hq={q.size(1)} not divisible by Hkv={k.size(1)} (GQA group)")


def _validate_scale(scale: Optional[float]) -> None:
    if scale is None:
        return
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError("scale must be a positive finite number")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("scale must be a positive finite number")


def _validate_output_dtype(output_dtype: torch.dtype) -> None:
    if not isinstance(output_dtype, torch.dtype):
        raise ValueError("output_dtype must be a real floating-point torch dtype")
    probe = torch.empty((), dtype=output_dtype)
    if not torch.is_floating_point(probe) or torch.is_complex(probe):
        raise ValueError("output_dtype must be a real floating-point torch dtype")


def _zero_dependency(*tensors: torch.Tensor) -> torch.Tensor:
    total = torch.tensor(0.0, device=tensors[0].device)
    for tensor in tensors:
        total = total + tensor.sum()
    return total * 0.0


def _normalize_position_offsets(
    offsets: Optional[torch.Tensor],
    batch: int,
    device: torch.device,
    *,
    default: int,
    name: str,
) -> torch.Tensor:
    if offsets is None:
        return torch.full((batch,), default, dtype=torch.long, device=device)
    if offsets.ndim != 1 or offsets.numel() != batch:
        raise ValueError(f"{name} must have shape [B]")
    if torch.is_floating_point(offsets) or torch.is_complex(offsets) or offsets.dtype == torch.bool:
        raise ValueError(f"{name} must contain integer positions")
    return offsets.to(device=device, dtype=torch.long)


def _split_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(length, parts)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        width = base + (1 if index < extra else 0)
        end = start + width
        bounds.append((start, end))
        start = end
    return bounds


def _kv_block_bounds(
    length: int,
    cp_world_size: int,
    kv_chunk_size: Optional[int],
) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for start, end in _split_bounds(length, cp_world_size):
        if kv_chunk_size is None:
            bounds.append((start, end))
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + kv_chunk_size, end)
            bounds.append((cursor, chunk_end))
            cursor = chunk_end
    return bounds


def split_kv_execution_plan_provenance(
    length: int,
    *,
    cp_world_size: int,
    kv_chunk_size: Optional[int],
    backend: str,
) -> list[dict[str, object]]:
    """Return the actual backend-local Split-KV plan for every CP owner."""

    if length < 1:
        raise ValueError("Split-KV sequence length must be >= 1")
    if cp_world_size < 1:
        raise ValueError("cp_world_size must be >= 1")
    if kv_chunk_size is not None and kv_chunk_size < 1:
        raise ValueError("kv_chunk_size must be >= 1 when provided")
    result: list[dict[str, object]] = []
    for owner_cp_rank, (rank_start, rank_end) in enumerate(_split_bounds(length, cp_world_size)):
        if rank_start == rank_end:
            continue
        boundaries: tuple[tuple[int, int], ...]
        if kv_chunk_size is None:
            boundaries = ((rank_start, rank_end),)
            mode = SplitKVMode.DISABLED
        else:
            boundaries = tuple(
                (start, min(start + kv_chunk_size, rank_end))
                for start in range(rank_start, rank_end, kv_chunk_size)
            )
            mode = SplitKVMode.FIXED
        plan = SplitKVExecutionPlan(
            requested_mode=mode,
            requested_split_size=kv_chunk_size,
            actual_mode=mode,
            actual_split_size=kv_chunk_size,
            boundaries=boundaries,
            backend=backend,
            source="reference_execution",
        )
        result.append({"owner_cp_rank": owner_cp_rank, **plan.to_dict()})
    return result


def build_reference_split_kv_runtime_plan_set(
    total_kv_tokens: Sequence[int],
    *,
    tp_world_size: int,
    cp_world_size: int,
    kv_chunk_size: Optional[int],
    backend: str = "deterministic_cp_reference",
) -> SplitKVRuntimePlanSet:
    """Build complete per-batch/TP/CP/owner plans for the reference path."""

    totals = tuple(total_kv_tokens)
    if not totals or any(total < cp_world_size for total in totals):
        raise ValueError("reference runtime plan sets require at least one KV token per CP owner")
    if tp_world_size < 1 or cp_world_size < 1:
        raise ValueError("TP and CP world sizes must be >= 1")
    if kv_chunk_size is not None and kv_chunk_size < 1:
        raise ValueError("kv_chunk_size must be >= 1 when provided")

    entries: list[SplitKVRuntimePlanEntry] = []
    for batch_index, total in enumerate(totals):
        owner_ranges = _split_bounds(total, cp_world_size)
        for tp_rank in range(tp_world_size):
            for cp_rank in range(cp_world_size):
                for owner_cp_rank, (owner_start, owner_end) in enumerate(owner_ranges):
                    boundaries: tuple[tuple[int, int], ...]
                    if kv_chunk_size is None:
                        mode = SplitKVMode.DISABLED
                        boundaries = ((owner_start, owner_end),)
                    else:
                        mode = SplitKVMode.FIXED
                        boundaries = tuple(
                            (start, min(start + kv_chunk_size, owner_end))
                            for start in range(owner_start, owner_end, kv_chunk_size)
                        )
                    execution = SplitKVExecutionPlan(
                        requested_mode=mode,
                        requested_split_size=kv_chunk_size,
                        actual_mode=mode,
                        actual_split_size=kv_chunk_size,
                        boundaries=boundaries,
                        backend=backend,
                        source="reference_execution",
                    )
                    entries.append(
                        SplitKVRuntimePlanEntry(
                            coordinate=SplitKVRuntimeCoordinate(
                                batch_index=batch_index,
                                tp_rank=tp_rank,
                                cp_rank=cp_rank,
                                owner_cp_rank=owner_cp_rank,
                            ),
                            expected_kv_range=(owner_start, owner_end),
                            execution=execution,
                        )
                    )
    return SplitKVRuntimePlanSet(
        batch_size=len(totals),
        tp_world_size=tp_world_size,
        cp_world_size=cp_world_size,
        total_kv_tokens=totals,
        entries=tuple(entries),
    )


CPAttentionReferenceOp = DeterministicCPAttentionReferenceOp

__all__ = [
    "AttentionBackwardComparisonReport",
    "AttentionBackwardGradients",
    "AttentionBackwardPathDrift",
    "AttentionBackwardPathResult",
    "AttentionBackwardRankDrift",
    "AttentionPartialState",
    "build_reference_split_kv_runtime_plan_set",
    "CPAttentionReferenceOp",
    "DeterministicCPAttentionReferenceOp",
    "GradientDriftStats",
    "compare_cp_attention_backward",
    "merge_attention_partial_states",
    "split_kv_execution_plan_provenance",
]
