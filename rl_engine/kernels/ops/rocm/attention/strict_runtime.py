# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework-neutral strict ROCm Attention runtime.

Composes the production AITER/CK core with the self-owned RCCL AG/RS
transport, mirroring :class:`StrictCUDAAttentionRuntime` so both platforms
present one runtime shape to framework integrations. Before this existed the
CP schedule had no home on ROCm: the core is single-rank arithmetic, the Vime
provider fails closed at ``CP > 1``, and the only working AG/core/RS sequence
lived in the benchmark script.

Two things differ from the CUDA runtime and both are load-bearing:

* The core is launched once per ``(batch row, KV group)`` rather than once per
  sequence. AITER/CK's reduction order depends on how many heads shared the
  launch, so a head shard computed under TP=N is otherwise not bit-identical
  to the same shard under a different TP degree. The CUDA FA4 core has no such
  dependence and runs one launch per sequence.
* RCCL moves tensors but never reduces them. The cross-rank ``(out, lse)``
  combine order comes from the fixed balanced rank tree in
  ``_RCCLRankOrderedTransport``, not from RCCL's own algorithm selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    AttentionContract,
)
from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionParallelSpec,
    RCCLAGRSAttentionCPCommunication,
)
from rl_engine.kernels.ops.cuda.attention.strict_runtime import StrictCUDAAttentionRuntime
from rl_engine.kernels.ops.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore

# The sequence reorder and the position validation are platform-neutral tensor
# bookkeeping. They are bound from the CUDA runtime rather than reimplemented
# so the two runtimes cannot drift into two different global orderings.
_sort_by_position = StrictCUDAAttentionRuntime._sort_by_position
_gather_sequence = StrictCUDAAttentionRuntime._gather_sequence
_validate_local_positions = StrictCUDAAttentionRuntime._validate_local_positions
_validate_global_positions = StrictCUDAAttentionRuntime._validate_global_positions


@dataclass(frozen=True)
class StrictRocmAttentionResult:
    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]


class StrictRocmAttentionRuntime:
    """Run one AITER/CK arithmetic identity at CP=1 or through RCCL AG/RS."""

    backend_id = "rlkernel.rocm.attention.aiter_ck_ag_rs.v1"
    core_id = STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_ROCM_SCHEDULE_ID
    communication_backend_id = "rccl_ag_rs"
    # Communication and compute are deliberately decoupled; see
    # ``AttentionCPCommunicationPlan.validate``.
    supports_async_overlap = False
    supports_compute_communication_fusion = False

    def __init__(
        self,
        *,
        process_group: Any = None,
        core: Any | None = None,
        communication: Any | None = None,
    ) -> None:
        self._core = StrictRocmAiterCKAttentionCore() if core is None else core
        self._communication = (
            RCCLAGRSAttentionCPCommunication(process_group=process_group)
            if communication is None
            else communication
        )
        if getattr(self._core, "core_id", None) != self.core_id:
            raise RuntimeError(
                "strict ROCm Attention runtime requires the AITER/CK production core"
            )
        if getattr(self._core, "strict_schedule", None) != self.strict_schedule:
            raise RuntimeError("strict ROCm Attention runtime requires the AITER/CK fixed schedule")
        self.communication_executed = False

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        contract: AttentionContract,
        causal: bool,
        scale: float | None,
        cp_world_size: int,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
        positions_are_sorted: bool = False,
    ) -> StrictRocmAttentionResult:
        self._require_rocm(q)
        if cp_world_size != contract.sharding.cp_world_size:
            raise RuntimeError("runtime CP world size does not match AttentionContract")
        _validate_local_positions(q, k, query_position_ids, key_position_ids)

        plan = None
        if cp_world_size == 1:
            global_q, global_k, global_v = q, k, v
            global_q_positions = query_position_ids
            global_k_positions = key_position_ids
            communication_backend = "none"
            self.communication_executed = False
        else:
            plan = self._communication_plan(contract, q.size(2), k.size(2))
            global_q = self._communication.all_gather_query(q, plan)
            global_k, global_v = self._communication.all_gather_kv(k, v, plan)
            global_q_positions, global_k_positions = self._communication.all_gather_position_ids(
                query_position_ids,
                key_position_ids,
                plan,
            )
            communication_backend = self.communication_backend_id
            self.communication_executed = True

        if positions_are_sorted:
            if cp_world_size != 1:
                raise RuntimeError("pre-sorted Attention positions are supported only at CP=1")
            q_sorted, k_sorted, v_sorted = global_q, global_k, global_v
            q_positions_sorted, k_positions_sorted = global_q_positions, global_k_positions
            q_sort = None
        else:
            q_sorted, q_positions_sorted, q_sort = _sort_by_position(global_q, global_q_positions)
            k_sorted, k_positions_sorted, k_sort = _sort_by_position(global_k, global_k_positions)
            v_sorted = _gather_sequence(global_v, k_sort)
            _validate_global_positions(q_positions_sorted, k_positions_sorted, causal)

        out_sorted, lse_sorted, core_provenance, launches = self._run_core(
            q_sorted,
            k_sorted,
            v_sorted,
            causal=causal,
            scale=scale,
            query_position_ids=q_positions_sorted,
            key_position_ids=k_positions_sorted,
            output_dtype=q.dtype,
        )

        if cp_world_size > 1:
            if q_sort is None:
                raise RuntimeError("CP Attention requires a framework position reorder")
            inverse_q_sort = torch.argsort(q_sort, dim=1)
            out_rank_packed = _gather_sequence(out_sorted, inverse_q_sort)
            lse_rank_packed = _gather_sequence(lse_sorted, inverse_q_sort)
            shard = self._communication.reduce_scatter_strict_result(
                out_rank_packed,
                lse_rank_packed,
                plan,
            )
            out, lse = shard.out, shard.lse
        else:
            out, lse = out_sorted, lse_sorted

        backend = (
            core_provenance.get("attention_backend")
            or core_provenance.get("actual_backend")
            or getattr(self._core, "backend_id", None)
        )
        return StrictRocmAttentionResult(
            out=out,
            lse=lse,
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "actual_backend": self.backend_id,
                "communication_backend": communication_backend,
                "communication_executed": self.communication_executed,
                "native_attention_arithmetic": True,
                "production_ready": True,
                "fallback": False,
                "fallback_reason": None,
                "reference_only": False,
                "split_kv": "disabled",
                "framework_position_reorder": True,
                # Unlike the CUDA runtime's single full-sequence launch, the
                # query schedule here is one launch per (batch row, KV group).
                "query_schedule": "one_batch_row_one_kv_group",
                "backward_schedule": "aiter_ck_deterministic_per_kv_group",
                "launch_granularity": "one_batch_row_one_kv_group",
                "tp_degree_invariant": True,
                "invariance_mechanism": "one_kv_group_per_launch",
                "core_row_count": q_sorted.size(0) * q_sorted.size(2),
                "core_launch_count": launches,
                "core_batch_size": q_sorted.size(0),
                "core_query_length": q_sorted.size(2),
                "core_actual_backends": [] if backend is None else [str(backend)],
                "core": core_provenance,
            },
        )

    def _run_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float | None,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Launch the core once per ``(batch row, KV group)`` and concatenate.

        Every launch therefore sees exactly one KV group and its Q heads, so
        the result does not depend on the TP degree that produced the shard.
        """

        local_kv_heads = k.size(1)
        if local_kv_heads <= 0 or q.size(1) % local_kv_heads:
            raise RuntimeError(
                f"local Q heads={q.size(1)} must be divisible by local KV heads={local_kv_heads}"
            )
        group_size = q.size(1) // local_kv_heads

        row_outs: list[torch.Tensor] = []
        row_lses: list[torch.Tensor] = []
        core_provenance: dict[str, Any] | None = None
        launches = 0
        for row in range(q.size(0)):
            row_query_positions = query_position_ids[row : row + 1]
            row_key_positions = key_position_ids[row : row + 1]
            group_outs: list[torch.Tensor] = []
            group_lses: list[torch.Tensor] = []
            for group in range(local_kv_heads):
                q_lo, q_hi = group * group_size, (group + 1) * group_size
                result = self._core.forward_with_lse(
                    q[row : row + 1, q_lo:q_hi],
                    k[row : row + 1, group : group + 1],
                    v[row : row + 1, group : group + 1],
                    causal=causal,
                    scale=scale,
                    key_padding_mask=None,
                    query_position_ids=row_query_positions if causal else None,
                    key_position_ids=row_key_positions if causal else None,
                    output_dtype=output_dtype,
                )
                group_outs.append(result.out)
                group_lses.append(result.lse)
                launches += 1
                if core_provenance is None:
                    core_provenance = dict(result.provenance)
            row_outs.append(torch.cat(group_outs, dim=1))
            row_lses.append(torch.cat(group_lses, dim=1))

        if core_provenance is None:
            raise RuntimeError("strict ROCm Attention runtime executed no core launch")
        return (
            torch.cat(row_outs, dim=0),
            torch.cat(row_lses, dim=0),
            core_provenance,
            launches,
        )

    @staticmethod
    def _require_rocm(tensor: torch.Tensor) -> None:
        if tensor.device.type != "cuda" or torch.version.hip is None:
            raise RuntimeError("strict ROCm Attention requires ROCm GPU tensors")

    @staticmethod
    def _communication_plan(
        contract: AttentionContract,
        local_q_tokens: int,
        local_kv_tokens: int,
    ) -> AttentionCPCommunicationPlan:
        sharding = contract.sharding
        parallel = AttentionParallelSpec(
            tp_world_size=sharding.tp_world_size,
            tp_rank=sharding.tp_rank,
            cp_world_size=sharding.cp_world_size,
            cp_rank=sharding.cp_rank,
        )
        query_ranges = tuple(
            (rank * local_q_tokens, (rank + 1) * local_q_tokens)
            for rank in range(sharding.cp_world_size)
        )
        blocks = tuple(
            AttentionCPBlockMetadata(
                global_block_index=rank,
                kv_block_start=rank * local_kv_tokens,
                kv_block_end=(rank + 1) * local_kv_tokens,
                owner_cp_rank=rank,
                owner_tp_rank=sharding.tp_rank,
            )
            for rank in range(sharding.cp_world_size)
        )
        return AttentionCPCommunicationPlan(
            parallel=parallel,
            backend="rccl_ag_rs",
            status="implemented",
            expected_blocks=blocks,
            expected_kv_token_range=(0, local_kv_tokens * sharding.cp_world_size),
            query_token_ranges=query_ranges,
        )


__all__ = ["StrictRocmAttentionResult", "StrictRocmAttentionRuntime"]
