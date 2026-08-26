# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Attention references and strict runtime-provenance normalization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Mapping, Sequence

from rl_engine.mismatch.schema import (
    CollectiveContract,
    CollectiveOp,
    DeterminismLevel,
    DowncastPoint,
    ExecutionPath,
    LibraryPin,
    ParallelDim,
    Precision,
    ReductionOrder,
    ReferenceAuthority,
    ReferenceImplementation,
    RequiredSetting,
    SettingChannel,
)

ATTENTION_LSE_DOMAIN = "attention"
ATTENTION_MERGE_STATE = "out_lse"
SPLIT_KV_PLAN_EVIDENCE = "split_kv_runtime_plan_set"
CP_BLOCK_MANIFEST_EVIDENCE = "cp_block_manifest"
ATTENTION_LSE_EVIDENCE = "attention_lse_export"
POST_ROPE_QK_EVIDENCE = "post_rope_qk_digest"

DTYPES: dict[str, Precision] = {
    "bf16": Precision.BF16,
    "bfloat16": Precision.BF16,
    "fp16": Precision.FP16,
    "float16": Precision.FP16,
    "fp32": Precision.FP32,
    "float32": Precision.FP32,
}

DOWNCAST_POINTS: dict[str, DowncastPoint] = {
    "never": DowncastPoint.NEVER,
    "per_block": DowncastPoint.PER_BLOCK,
    "per_partial": DowncastPoint.PER_PARTIAL,
    "final_write": DowncastPoint.FINAL_WRITE,
}

COLLECTIVE_OPS: dict[str, CollectiveOp] = {item.value: item for item in CollectiveOp}
REDUCTION_ORDERS: dict[str, ReductionOrder] = {item.value: item for item in ReductionOrder}
DETERMINISM_LEVELS: dict[str, DeterminismLevel] = {
    item.value: item for item in DeterminismLevel
}


class AttentionContractError(ValueError):
    """Raised when runtime metadata cannot prove an Attention contract."""


REFERENCE_CP_MERGE = CollectiveContract(
    op=CollectiveOp.POINT_TO_POINT,
    group=ParallelDim.CONTEXT,
    group_size=2,
    reduction_order=ReductionOrder.GLOBAL_BLOCK_INDEX,
    accumulate_precision=Precision.FP32,
    downcast_at=DowncastPoint.FINAL_WRITE,
    determinism=DeterminismLevel.STABLE_ACROSS_TOPOLOGY,
    backend="p2p_nccl_reference",
)

TE_ROPE_REFERENCE = ReferenceImplementation(
    name="transformer_engine",
    tier=ReferenceAuthority.SHARED_BACKEND,
    training_impl="transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb",
    rollout_impl="flashinfer.rope.apply_rope",
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
    ),
    required_settings=(
        RequiredSetting(
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            "0",
            SettingChannel.ENV_VAR,
            readback="os.environ",
        ),
    ),
    pinned_libraries=(LibraryPin("transformer_engine", "2.9.0.dev0", commit="8260f49"),),
)

SPLIT_KV_REFERENCE = ReferenceImplementation(
    name="rl_kernel",
    tier=ReferenceAuthority.SELF_WRITTEN,
    training_impl=(
        "rl_engine.kernels.ops.pytorch.attention.cp_attention."
        "DeterministicCPAttentionReferenceOp"
    ),
    rollout_impl=(
        "rl_engine.kernels.ops.pytorch.attention.cp_attention."
        "DeterministicCPAttentionReferenceOp"
    ),
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
        ExecutionPath.ROLLOUT_CHUNKED_PREFILL,
    ),
    pinned_libraries=(LibraryPin("torch", "2.6.0"),),
)

CP_MERGE_REFERENCE = ReferenceImplementation(
    name="p2p_nccl_reference",
    tier=ReferenceAuthority.SELF_WRITTEN,
    training_impl=(
        "rl_engine.kernels.ops.cuda.attention.cp_comm.P2PNCCLAttentionCPCommunication"
    ),
    rollout_impl=(
        "rl_engine.kernels.ops.cuda.attention.cp_comm.P2PNCCLAttentionCPCommunication"
    ),
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
        ExecutionPath.ROLLOUT_CHUNKED_PREFILL,
    ),
    required_settings=(
        RequiredSetting(
            "attn.cp_collective",
            REFERENCE_CP_MERGE,
            SettingChannel.CALL_ARG,
            readback="dispatch.provenance['cp_collective']",
        ),
    ),
    pinned_libraries=(LibraryPin("nccl", "2.21.5"),),
)


def precision(value: Any, field: str) -> Precision:
    if isinstance(value, Precision):
        return value
    key = str(value).lower()
    if key not in DTYPES:
        raise AttentionContractError(f"{field} must be one of {tuple(DTYPES)}, got {value!r}")
    return DTYPES[key]


def downcast_point(value: Any, field: str) -> DowncastPoint:
    if isinstance(value, DowncastPoint):
        return value
    key = str(value).lower()
    if key not in DOWNCAST_POINTS:
        raise AttentionContractError(
            f"{field} must be one of {tuple(DOWNCAST_POINTS)}, got {value!r}"
        )
    return DOWNCAST_POINTS[key]


def positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise AttentionContractError(f"{field} must be a positive integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AttentionContractError(
            f"{field} must be a positive integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise AttentionContractError(f"{field} must be a positive integer, got {value!r}")
    return parsed


def optional_positive_int(value: Any, field: str) -> int | None:
    return None if value is None else positive_int(value, field)


def reference_selected(value: Any, role: Any, reference_name: str, field: str) -> bool:
    if value in (None, "native"):
        return False
    if value == reference_name:
        return True
    role_value = getattr(role, "value", str(role))
    if value in (f"{reference_name}@training", f"{reference_name}@rollout"):
        return value.endswith(f"@{role_value}")
    raise AttentionContractError(
        f"unknown {field} value {value!r}; expected 'native', {reference_name!r}, "
        f"'{reference_name}@training' or '{reference_name}@rollout'"
    )


def normalize_collective(
    raw: Any,
    *,
    cp_world_size: int,
) -> tuple[CollectiveContract, ...]:
    """Normalize the collective that actually ran; absence stays unknown."""

    if raw is None:
        return ()
    if isinstance(raw, CollectiveContract):
        contract = raw
    elif isinstance(raw, Mapping):
        op = _enum_member(COLLECTIVE_OPS, raw.get("op"), "attn.cp_collective.op")
        order = _enum_member(
            REDUCTION_ORDERS,
            raw.get("reduction_order"),
            "attn.cp_collective.reduction_order",
        )
        determinism = _enum_member(
            DETERMINISM_LEVELS,
            raw.get("determinism"),
            "attn.cp_collective.determinism",
        )
        contract = CollectiveContract(
            op=op,
            group=ParallelDim.CONTEXT,
            group_size=positive_int(
                raw.get("group_size", cp_world_size), "attn.cp_collective.group_size"
            ),
            reduction_order=order,
            accumulate_precision=precision(
                raw.get("accumulate_precision"),
                "attn.cp_collective.accumulate_precision",
            ),
            downcast_at=downcast_point(
                raw.get("downcast_at"), "attn.cp_collective.downcast_at"
            ),
            determinism=determinism,
            backend=_non_empty_string(raw.get("backend"), "attn.cp_collective.backend"),
        )
    else:
        raise AttentionContractError(
            "attn.cp_collective must be a CollectiveContract or mapping"
        )

    if contract.group is not ParallelDim.CONTEXT:
        raise AttentionContractError("Attention CP collective must use the context group")
    if contract.group_size != cp_world_size:
        raise AttentionContractError(
            "Attention CP collective group_size must equal attn.cp_world_size"
        )
    if contract.accumulate_precision is not Precision.FP32:
        raise AttentionContractError("Attention (out, lse) merge must accumulate in fp32")
    return (contract,)


def reference_collective(cp_world_size: int) -> tuple[CollectiveContract, ...]:
    return (replace(REFERENCE_CP_MERGE, group_size=cp_world_size),)


def normalize_split_kv_plan_set(raw: Any) -> Mapping[str, Any] | None:
    """Validate and canonicalize complete batch/TP/CP/owner runtime plans.

    A requested policy is deliberately not accepted here. Every coordinate must
    carry the actual logical boundaries and numerical merge semantics.
    """

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise AttentionContractError("attn.actual_split_kv_plan_set must be a mapping")

    batch_size = positive_int(raw.get("batch_size"), "split_kv.batch_size")
    tp_world_size = positive_int(raw.get("tp_world_size"), "split_kv.tp_world_size")
    cp_world_size = positive_int(raw.get("cp_world_size"), "split_kv.cp_world_size")
    totals = _int_tuple(raw.get("total_kv_tokens"), "split_kv.total_kv_tokens")
    if len(totals) != batch_size or any(value <= 0 for value in totals):
        raise AttentionContractError(
            "split_kv.total_kv_tokens must contain one positive length per batch item"
        )

    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, Sequence) or isinstance(entries_raw, (str, bytes)):
        raise AttentionContractError("split_kv.entries must be a sequence")

    expected_coordinates = {
        (batch, tp, cp, owner)
        for batch in range(batch_size)
        for tp in range(tp_world_size)
        for cp in range(cp_world_size)
        for owner in range(cp_world_size)
    }
    entries: list[tuple[Any, ...]] = []
    seen: set[tuple[int, int, int, int]] = set()
    by_owner: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)

    for index, entry_raw in enumerate(entries_raw):
        if not isinstance(entry_raw, Mapping):
            raise AttentionContractError(f"split_kv.entries[{index}] must be a mapping")
        coordinate = (
            _bounded_int(entry_raw.get("batch_index"), batch_size, "batch_index"),
            _bounded_int(entry_raw.get("tp_rank"), tp_world_size, "tp_rank"),
            _bounded_int(entry_raw.get("cp_rank"), cp_world_size, "cp_rank"),
            _bounded_int(entry_raw.get("owner_cp_rank"), cp_world_size, "owner_cp_rank"),
        )
        if coordinate in seen:
            raise AttentionContractError(f"duplicate Split-KV coordinate {coordinate}")
        seen.add(coordinate)

        expected_range = _range_pair(
            entry_raw.get("expected_kv_range"),
            f"split_kv.entries[{index}].expected_kv_range",
        )
        if expected_range[1] > totals[coordinate[0]]:
            raise AttentionContractError("Split-KV expected range exceeds total_kv_tokens")

        requested_mode = _mode(entry_raw.get("requested_split_kv_policy"), "requested mode")
        actual_mode = _mode(entry_raw.get("actual_split_kv_policy"), "actual mode")
        requested_size = optional_positive_int(
            entry_raw.get("requested_split_kv_size"), "requested_split_kv_size"
        )
        actual_size = optional_positive_int(
            entry_raw.get("actual_split_kv_size"), "actual_split_kv_size"
        )
        boundaries = _boundaries(
            entry_raw.get("actual_split_boundaries"), expected_range, index
        )
        reported_count = entry_raw.get("actual_split_kv_count")
        if reported_count is not None and positive_int(
            reported_count, "actual_split_kv_count"
        ) != len(boundaries):
            raise AttentionContractError(
                "actual_split_kv_count must equal the number of actual boundaries"
            )
        merge_order = _enum_member(
            REDUCTION_ORDERS,
            entry_raw.get("split_kv_merge_order"),
            "split_kv_merge_order",
        )
        accumulate = precision(
            entry_raw.get("split_kv_accum_dtype"), "split_kv_accum_dtype"
        )
        downcast = downcast_point(
            entry_raw.get("split_kv_downcast_at"), "split_kv_downcast_at"
        )
        backend = _non_empty_string(
            entry_raw.get("split_kv_backend"), "split_kv_backend"
        )
        source = _non_empty_string(
            entry_raw.get("split_kv_plan_source"), "split_kv_plan_source"
        )
        fallback = entry_raw.get("split_kv_fallback")
        if not isinstance(fallback, bool):
            raise AttentionContractError("split_kv_fallback must be a bool")
        fallback_reason = entry_raw.get("split_kv_fallback_reason")
        if fallback and (not isinstance(fallback_reason, str) or not fallback_reason.strip()):
            raise AttentionContractError(
                "a Split-KV fallback must include a non-empty fallback_reason"
            )
        if not fallback and fallback_reason is not None:
            raise AttentionContractError(
                "split_kv_fallback_reason must be None when fallback is false"
            )

        _validate_split_mode_sizes(
            requested_mode,
            requested_size,
            actual_mode,
            actual_size,
            boundaries,
            fallback,
        )
        if merge_order is not ReductionOrder.GLOBAL_BLOCK_INDEX:
            raise AttentionContractError(
                "Split-KV partials must merge in global_block_index order"
            )
        if accumulate is not Precision.FP32:
            raise AttentionContractError("Split-KV partials must accumulate in fp32")
        if downcast is not DowncastPoint.FINAL_WRITE:
            raise AttentionContractError("Split-KV partials may downcast only at final_write")

        canonical = (
            *coordinate,
            expected_range,
            requested_mode,
            requested_size,
            actual_mode,
            actual_size,
            boundaries,
            merge_order,
            accumulate,
            downcast,
            backend,
            source,
            fallback,
            fallback_reason,
        )
        entries.append(canonical)
        by_owner[(coordinate[0], coordinate[3])].append(canonical)

    missing = expected_coordinates - seen
    extra = seen - expected_coordinates
    if missing or extra:
        raise AttentionContractError(
            "Split-KV runtime plan coverage is incomplete; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    _validate_owner_ranges(entries, batch_size, tp_world_size, cp_world_size, totals)
    for owner, owner_entries in by_owner.items():
        reference = _plan_semantics(owner_entries[0])
        if any(_plan_semantics(entry) != reference for entry in owner_entries[1:]):
            raise AttentionContractError(
                "Split-KV plan differs across TP/CP consumers for "
                f"batch={owner[0]}, owner_cp={owner[1]}"
            )

    ordered = tuple(sorted(entries, key=lambda item: item[:4]))
    return {
        "batch_size": batch_size,
        "tp_world_size": tp_world_size,
        "cp_world_size": cp_world_size,
        "total_kv_tokens": totals,
        "coordinates": tuple(item[:4] for item in ordered),
        "owner_ranges": tuple((item[:4], item[4]) for item in ordered),
        "boundaries": tuple((item[:4], item[9]) for item in ordered),
        "merge_order": tuple((item[:4], item[10]) for item in ordered),
        "accumulate_precision": tuple((item[:4], item[11]) for item in ordered),
        "downcast_at": tuple((item[:4], item[12]) for item in ordered),
        "fallback": tuple((item[:4], item[15], item[16]) for item in ordered),
        "backend": tuple((item[:4], item[13]) for item in ordered),
        "source": tuple((item[:4], item[14]) for item in ordered),
        "canonical": tuple(_cross_side_plan_semantics(item) for item in ordered),
    }


def normalize_cp_block_manifest(
    raw: Any,
    *,
    tp_world_size: int,
    cp_world_size: int,
) -> tuple[tuple[int, int, int, int, int], ...] | None:
    """Validate global block ownership and gap-free KV range coverage."""

    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise AttentionContractError("attn.cp_block_manifest must be a non-empty sequence")

    blocks: list[tuple[int, int, int, int, int]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise AttentionContractError(f"attn.cp_block_manifest[{index}] must be a mapping")
        block_index = _non_negative_int(item.get("global_block_index"), "global_block_index")
        start, end = _range_pair(
            (item.get("kv_block_start"), item.get("kv_block_end")),
            "KV block range",
        )
        owner_cp = _bounded_int(item.get("owner_cp_rank"), cp_world_size, "owner_cp_rank")
        owner_tp = _bounded_int(item.get("owner_tp_rank"), tp_world_size, "owner_tp_rank")
        blocks.append((block_index, start, end, owner_cp, owner_tp))

    ordered = tuple(sorted(blocks))
    if len({block[0] for block in ordered}) != len(ordered):
        raise AttentionContractError("CP block manifest contains duplicate global_block_index")
    if tuple(block[0] for block in ordered) != tuple(range(len(ordered))):
        raise AttentionContractError("CP global_block_index values must be contiguous from zero")
    previous_end = ordered[0][1]
    for _, start, end, _, _ in ordered:
        if start != previous_end:
            raise AttentionContractError("CP block KV ranges must be gap-free and non-overlapping")
        previous_end = end
    return ordered


def _enum_member(values: Mapping[str, Any], value: Any, field: str) -> Any:
    key = value.value if hasattr(value, "value") else str(value).lower()
    if key not in values:
        raise AttentionContractError(f"{field} must be one of {tuple(values)}, got {value!r}")
    return values[key]


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AttentionContractError(f"{field} must be a non-empty string")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AttentionContractError(f"{field} must be a non-negative integer")
    return value


def _bounded_int(value: Any, upper: int, field: str) -> int:
    parsed = _non_negative_int(value, field)
    if parsed >= upper:
        raise AttentionContractError(f"{field}={parsed} must be smaller than {upper}")
    return parsed


def _int_tuple(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AttentionContractError(f"{field} must be a sequence of integers")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise AttentionContractError(f"{field} must be a sequence of integers")
    return result


def _range_pair(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise AttentionContractError(f"{field} must be a (start, end) pair")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise AttentionContractError(f"{field} must satisfy 0 <= start < end")
    return start, end


def _boundaries(
    raw: Any,
    expected_range: tuple[int, int],
    entry_index: int,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise AttentionContractError(
            f"split_kv.entries[{entry_index}] must report actual_split_boundaries"
        )
    boundaries = tuple(
        _range_pair(item, f"split_kv.entries[{entry_index}].actual_split_boundaries")
        for item in raw
    )
    previous_end = expected_range[0]
    for start, end in boundaries:
        if start != previous_end:
            raise AttentionContractError(
                "actual Split-KV boundaries must be gap-free in logical KV order"
            )
        previous_end = end
    if previous_end != expected_range[1]:
        raise AttentionContractError(
            "actual Split-KV boundaries must exactly cover expected_kv_range"
        )
    return boundaries


def _mode(value: Any, field: str) -> str:
    if value not in ("disabled", "fixed", "auto"):
        raise AttentionContractError(
            f"{field} must be 'disabled', 'fixed', or 'auto', got {value!r}"
        )
    return value


def _validate_split_mode_sizes(
    requested_mode: str,
    requested_size: int | None,
    actual_mode: str,
    actual_size: int | None,
    boundaries: tuple[tuple[int, int], ...],
    fallback: bool,
) -> None:
    if (requested_mode == "fixed") != (requested_size is not None):
        raise AttentionContractError(
            "requested fixed Split-KV mode and requested_split_kv_size must appear together"
        )
    if (actual_mode == "fixed") != (actual_size is not None):
        raise AttentionContractError(
            "actual fixed Split-KV mode and actual_split_kv_size must appear together"
        )
    if actual_mode == "disabled" and len(boundaries) != 1:
        raise AttentionContractError("disabled Split-KV must report exactly one boundary")
    if actual_mode == "fixed" and actual_size is not None:
        widths = tuple(end - start for start, end in boundaries)
        if any(width != actual_size for width in widths[:-1]) or widths[-1] > actual_size:
            raise AttentionContractError(
                "fixed Split-KV boundaries must use actual_split_kv_size except at the tail"
            )
    if not fallback and (requested_mode, requested_size) != (actual_mode, actual_size):
        raise AttentionContractError(
            "actual Split-KV mode/size may differ from the request only for a fallback"
        )


def _validate_owner_ranges(
    entries: Sequence[tuple[Any, ...]],
    batch_size: int,
    tp_world_size: int,
    cp_world_size: int,
    totals: Sequence[int],
) -> None:
    by_coordinate = {entry[:4]: entry for entry in entries}
    for batch in range(batch_size):
        for tp in range(tp_world_size):
            for cp in range(cp_world_size):
                previous_end = 0
                for owner in range(cp_world_size):
                    expected_range = by_coordinate[(batch, tp, cp, owner)][4]
                    if expected_range[0] != previous_end:
                        raise AttentionContractError(
                            "Split-KV owner ranges must be gap-free in CP owner order"
                        )
                    previous_end = expected_range[1]
                if previous_end != totals[batch]:
                    raise AttentionContractError(
                        "Split-KV owner ranges must cover total_kv_tokens"
                    )


def _plan_semantics(entry: tuple[Any, ...]) -> tuple[Any, ...]:
    # Drop TP/CP consumer coordinates and provenance labels. Within one side,
    # every consumer of an owner range must execute the same numerical plan.
    return (entry[0], entry[3], *entry[4:13], *entry[15:])


def _cross_side_plan_semantics(entry: tuple[Any, ...]) -> tuple[Any, ...]:
    # Backend and source are provenance, not numerical semantics.
    return (*entry[:13], *entry[15:])


__all__ = [
    "ATTENTION_LSE_DOMAIN",
    "ATTENTION_LSE_EVIDENCE",
    "ATTENTION_MERGE_STATE",
    "CP_BLOCK_MANIFEST_EVIDENCE",
    "CP_MERGE_REFERENCE",
    "DOWNCAST_POINTS",
    "POST_ROPE_QK_EVIDENCE",
    "REFERENCE_CP_MERGE",
    "SPLIT_KV_PLAN_EVIDENCE",
    "SPLIT_KV_REFERENCE",
    "TE_ROPE_REFERENCE",
    "AttentionContractError",
    "downcast_point",
    "normalize_collective",
    "normalize_cp_block_manifest",
    "normalize_split_kv_plan_set",
    "positive_int",
    "precision",
    "reference_collective",
    "reference_selected",
]
