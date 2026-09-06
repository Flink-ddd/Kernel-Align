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
  combine order comes from the fixed balanced rank tree in the shared
  ``RCCLDeterministicCollective``, not from RCCL's own algorithm selection.
  That is the collective the CUDA runtime also resolves through
  ``collective_for_group``, so both platforms run one reduction order from
  one implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    AttentionContract,
)
from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationUnavailable,
    AttentionCPCommunicationPlan,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
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


class RCCLAGRSAttentionCPCommunication(CUDAAGRSAttentionCPCommunication):
    """ROCm adapter over the shared deterministic fixed-tree collective."""

    backend_id = "rccl_ag_rs"
    collective_label = "self-owned RCCL AG/RS"
    supports_autograd = True
    transport_only = True
    supports_async_overlap = False
    supports_compute_communication_fusion = False

    def _dist(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires initialized torch.distributed"
            )
        return dist

    def _validate_cuda_plan(self, plan: AttentionCPCommunicationPlan) -> None:
        if plan.backend != "rccl_ag_rs" or plan.status != "implemented":
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires an implemented rccl_ag_rs plan"
            )
        replace(plan, backend="cuda_ag_rs").validate()
        if torch.version.hip is None or not torch.cuda.is_available():
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires an available ROCm device"
            )


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

    def forward_paged_with_lse(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        scale: float | None,
        out: torch.Tensor | None = None,
        cached_lengths: Sequence[int] | None = None,
    ) -> StrictRocmAttentionResult:
        """Run strict decode Attention over a paged KV cache.

        AITER exposes no paged entry point that this contract can use. Every
        ``paged_attention_*`` kernel partitions KV and reduces the partials
        (``partition_size``, ``exp_sums``/``max_logits``/``tmp_out``), so the
        partition count moves with the cached length; AITER's
        ``flash_attn_varlen_func`` takes a ``block_table`` but has no
        ``num_splits`` knob to pin, unlike CUDA's FA4. Either way the strict
        contract could not prove Split-KV disabled.

        So the pages are gathered into logical KV order and handed to the same
        dense core the prefill path uses, at the same one-launch-per
        ``(batch row, KV group)`` granularity. The arithmetic is then identical
        to a CP=1 prefill over the same logical sequence, which is what makes
        decode replay comparable against it. The cost is materializing the
        cached KV; a native paged kernel would avoid that, and can replace this
        once AITER can pin its split count.
        """

        self._require_rocm(q)
        self._validate_paged_inputs(
            q,
            k_cache,
            v_cache,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
        )
        if out is not None:
            if out.shape != q.shape:
                raise ValueError("paged Attention out must have the same shape as q")
            if out.dtype != q.dtype or out.device != q.device:
                raise ValueError("paged Attention out must match the Q dtype and device")
            if not out.is_contiguous():
                raise ValueError("paged Attention out must be contiguous")
        direct_core_out = (
            out is not None
            and not torch.is_grad_enabled()
            and not any(tensor.requires_grad for tensor in (q, k_cache, v_cache, out))
            and q.size(2) == 1
            and callable(getattr(self._core, "forward_decode_with_lse_into", None))
            and self._storage_is_disjoint(
                out,
                q,
                k_cache,
                v_cache,
                page_table,
                seqused_k,
            )
        )

        if cached_lengths is None:
            cached_lengths = tuple(int(value) for value in seqused_k.tolist())
        else:
            cached_lengths = tuple(int(value) for value in cached_lengths)
        if len(cached_lengths) != q.size(0):
            raise ValueError("cached_lengths must carry one length per query")
        for row, cached_length in enumerate(cached_lengths):
            if cached_length <= 0 or cached_length > max_seqlen_k:
                raise ValueError(
                    "seqused_k entries must be positive and within max_seqlen_k; "
                    f"row {row} requested {cached_length}"
                )

        row_outs: list[torch.Tensor] = []
        row_lses: list[torch.Tensor] = []
        core_provenance: dict[str, Any] | None = None
        launches = 0
        for row, cached_length in enumerate(cached_lengths):
            k_row, v_row = self._gather_paged_row(
                k_cache,
                v_cache,
                page_table[row],
                cached_length,
            )
            # Decode attends over the whole cached prefix, so neither the mask
            # nor the AITER call consumes position IDs. Avoid allocating two
            # dead tensors for every row of every decoder layer.
            row_out, row_lse, row_provenance, row_launches = self._run_core(
                q[row : row + 1],
                k_row,
                v_row,
                causal=False,
                scale=scale,
                query_position_ids=None,
                key_position_ids=None,
                output_dtype=q.dtype,
                out=None if out is None else out[row : row + 1],
                direct_core_out=direct_core_out,
            )
            if out is None:
                row_outs.append(row_out)
            row_lses.append(row_lse)
            launches += row_launches
            if core_provenance is None:
                core_provenance = row_provenance

        if core_provenance is None:
            raise RuntimeError("strict ROCm paged Attention executed no core launch")

        result_out = out if out is not None else torch.cat(row_outs, dim=0)
        result_lse = torch.cat(row_lses, dim=0)
        self.communication_executed = False

        backend = (
            core_provenance.get("attention_backend")
            or core_provenance.get("actual_backend")
            or getattr(self._core, "backend_id", None)
        )
        return StrictRocmAttentionResult(
            out=result_out,
            lse=result_lse,
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "actual_backend": self.backend_id,
                "communication_backend": "none",
                "communication_executed": False,
                "native_attention_arithmetic": True,
                "production_ready": True,
                "fallback": False,
                "fallback_reason": None,
                "reference_only": False,
                "split_kv": "disabled",
                "query_schedule": "paged_single_query_batch",
                # The dense core runs; the pages are gathered first. Recorded so
                # a reader never mistakes this for a native paged kernel.
                "paged_execution": "logical_kv_gather_then_dense_core",
                "paged_kernel": "none",
                "launch_granularity": "one_batch_row_one_kv_group",
                "tp_degree_invariant": True,
                "invariance_mechanism": "one_kv_group_per_launch",
                "core_row_count": q.size(0) * q.size(2),
                "core_launch_count": launches,
                "core_batch_size": q.size(0),
                "core_query_length": q.size(2),
                "core_actual_backends": [] if backend is None else [str(backend)],
                "core_output_staging": (
                    "aiter_direct_caller_group" if direct_core_out else "runtime_group_cat"
                ),
                "core": core_provenance,
            },
        )

    @staticmethod
    def _gather_paged_row(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_row: torch.Tensor,
        cached_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize one row's cached KV in logical order as ``[1, H, S, D]``.

        Physical page order never reaches the core: the pages are read through
        the page table, so the launch sees the same logical sequence a prefill
        over the same tokens would have seen.
        """

        page_size = k_cache.size(1)
        page_count = (cached_length + page_size - 1) // page_size
        if page_count > page_row.numel():
            raise ValueError("page_table row is shorter than the cached length requires")
        pages = page_row[:page_count]
        bounds_ok = torch.all((pages >= 0) & (pages < k_cache.size(0)))
        if pages.is_cuda:
            # Keep malformed metadata fail-closed without synchronizing the host
            # twice per row. vLLM page tables are already int32, which
            # index_select accepts directly on ROCm.
            torch._assert_async(bounds_ok, "page_table entries are outside the KV cache")
        elif not bool(bounds_ok.item()):
            raise ValueError("page_table entries are outside the KV cache")

        def _gather(cache: torch.Tensor) -> torch.Tensor:
            selected = cache.index_select(0, pages)
            flat = selected.reshape(page_count * page_size, cache.size(2), cache.size(3))
            return flat[:cached_length].permute(1, 0, 2).unsqueeze(0).contiguous()

        return _gather(k_cache), _gather(v_cache)

    @staticmethod
    def _validate_paged_inputs(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> None:
        if q.ndim != 4:
            raise ValueError("paged q must use [B, H, S, D]")
        if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
            raise ValueError("paged k/v must use [pages, page_size, H, D]")
        if q.size(1) % k_cache.size(2) != 0 or q.size(3) != k_cache.size(3):
            raise ValueError("paged q/k head counts or head dimensions are incompatible")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict paged Attention supports FP16/BF16 only")
        if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
            raise ValueError("paged q/k/v must share one dtype")
        if not (q.device == k_cache.device == v_cache.device):
            raise ValueError("paged q/k/v must be on one ROCm device")
        if page_table.ndim != 2 or page_table.size(0) != q.size(0):
            raise ValueError("page_table must be 2-D with one row per query")
        if page_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("page_table must be an integer tensor")
        if seqused_k.shape != (q.size(0),):
            raise ValueError("seqused_k must carry one cached length per query")
        if seqused_k.dtype not in (torch.int32, torch.int64):
            raise ValueError("seqused_k must be an integer tensor")
        if page_table.device != q.device or seqused_k.device != q.device:
            raise ValueError("paged Attention metadata must be on the Q device")
        if max_seqlen_k <= 0 or max_seqlen_k > page_table.size(1) * k_cache.size(1):
            raise ValueError("max_seqlen_k exceeds the page table capacity")

    def _run_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float | None,
        query_position_ids: torch.Tensor | None,
        key_position_ids: torch.Tensor | None,
        output_dtype: torch.dtype,
        out: torch.Tensor | None = None,
        direct_core_out: bool = False,
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
        if direct_core_out:
            if out is None or causal or q.size(2) != 1:
                raise RuntimeError("direct ROCm core output is only valid for paged decode")
            if torch.is_grad_enabled() or any(
                tensor.requires_grad for tensor in (q, k, v, out)
            ):
                raise RuntimeError("direct ROCm core output requires disabled gradient mode")
            if not callable(getattr(self._core, "forward_decode_with_lse_into", None)):
                raise RuntimeError("strict ROCm core has no direct decode output entry point")

        row_outs: list[torch.Tensor] = []
        row_lses: list[torch.Tensor] = []
        core_provenance: dict[str, Any] | None = None
        launches = 0
        for row in range(q.size(0)):
            row_query_positions = (
                None if query_position_ids is None else query_position_ids[row : row + 1]
            )
            row_key_positions = (
                None if key_position_ids is None else key_position_ids[row : row + 1]
            )
            group_outs: list[torch.Tensor] = []
            group_lses: list[torch.Tensor] = []
            for group in range(local_kv_heads):
                q_lo, q_hi = group * group_size, (group + 1) * group_size
                group_out = None
                if direct_core_out:
                    group_out = out[row : row + 1, q_lo:q_hi]
                    result = self._core.forward_decode_with_lse_into(
                        q[row : row + 1, q_lo:q_hi],
                        k[row : row + 1, group : group + 1],
                        v[row : row + 1, group : group + 1],
                        out=group_out,
                        scale=scale,
                        output_dtype=output_dtype,
                    )
                else:
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
                if group_out is None:
                    group_outs.append(result.out)
                elif result.out.data_ptr() != group_out.data_ptr():
                    raise RuntimeError("strict ROCm core did not write to its output slice")
                group_lses.append(result.lse)
                launches += 1
                if core_provenance is None:
                    core_provenance = dict(result.provenance)
            if out is None:
                row_outs.append(torch.cat(group_outs, dim=1))
            elif not direct_core_out:
                torch.cat(group_outs, dim=1, out=out[row : row + 1])
            row_lses.append(torch.cat(group_lses, dim=1))

        if core_provenance is None:
            raise RuntimeError("strict ROCm Attention runtime executed no core launch")
        return (
            out if out is not None else torch.cat(row_outs, dim=0),
            torch.cat(row_lses, dim=0),
            core_provenance,
            launches,
        )

    @staticmethod
    def _storage_is_disjoint(output: torch.Tensor, *inputs: torch.Tensor) -> bool:
        output_storage = output.untyped_storage().data_ptr()
        return all(tensor.untyped_storage().data_ptr() != output_storage for tensor in inputs)

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
