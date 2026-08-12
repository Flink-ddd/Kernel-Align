# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Attention adapter tests: actual Split-KV/CP evidence, not requested flags."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from rl_engine.mismatch.operator_checks.attention import AttentionChecks, adapter
from rl_engine.mismatch.operator_checks.attention._common import CP_MERGE_REFERENCE
from rl_engine.mismatch.schema import (
    CollectiveOp,
    DowncastPoint,
    PolicyRole,
    Precision,
    ReductionOrder,
)


def _plan_set(
    *,
    cp_world_size: int = 2,
    boundaries: dict[int, list[list[int]]] | None = None,
    actual_mode: str = "auto",
    actual_size: int | None = None,
    fallback: bool = False,
) -> dict:
    total = 8
    ranges = [(rank * 4, (rank + 1) * 4) for rank in range(cp_world_size)]
    if boundaries is None:
        boundaries = {
            owner: [[start, start + 2], [start + 2, end]]
            for owner, (start, end) in enumerate(ranges)
        }
    entries = []
    for cp_rank in range(cp_world_size):
        for owner, expected_range in enumerate(ranges):
            entries.append(
                {
                    "batch_index": 0,
                    "tp_rank": 0,
                    "cp_rank": cp_rank,
                    "owner_cp_rank": owner,
                    "expected_kv_range": list(expected_range),
                    "requested_split_kv_policy": "auto",
                    "requested_split_kv_size": None,
                    "actual_split_kv_policy": actual_mode,
                    "actual_split_kv_size": actual_size,
                    "actual_split_boundaries": boundaries[owner],
                    "split_kv_merge_order": "global_block_index",
                    "split_kv_accum_dtype": "fp32",
                    "split_kv_downcast_at": "final_write",
                    "split_kv_backend": "fixture",
                    "split_kv_plan_source": "runtime_trace",
                    "split_kv_fallback": fallback,
                    "split_kv_fallback_reason": "shape fallback" if fallback else None,
                }
            )
    return {
        "batch_size": 1,
        "tp_world_size": 1,
        "cp_world_size": cp_world_size,
        "total_kv_tokens": [total],
        "entries": entries,
    }


def _manifest() -> list[dict]:
    return [
        {
            "global_block_index": 0,
            "kv_block_start": 0,
            "kv_block_end": 4,
            "owner_cp_rank": 0,
            "owner_tp_rank": 0,
        },
        {
            "global_block_index": 1,
            "kv_block_start": 4,
            "kv_block_end": 8,
            "owner_cp_rank": 1,
            "owner_tp_rank": 0,
        },
    ]


def _collective(**overrides) -> dict:
    result = {
        "op": "point_to_point",
        "group_size": 2,
        "reduction_order": "global_block_index",
        "accumulate_precision": "fp32",
        "downcast_at": "final_write",
        "determinism": "stable_across_topology",
        "backend": "p2p_nccl_reference",
    }
    result.update(overrides)
    return result


def _effective(**overrides) -> dict:
    result = {
        "attn.compute_dtype": "bf16",
        "attn.accumulate_dtype": "fp32",
        "attn.downcast_at": "final_write",
        "attn.batch_size": 1,
        "attn.tp_world_size": 1,
        "attn.cp_world_size": 2,
        "attn.actual_split_kv_plan_set": _plan_set(),
        "attn.cp_block_manifest": _manifest(),
        "attn.cp_collective": _collective(),
        "attn.lse_domain": "attention",
        "attn.export_lse": True,
        "attn.merge_state": "out_lse",
        "attn.rope_theta": 1_000_000.0,
        "attn.position_ids_digest": "positions:abc",
        "attn.post_rope_qk_digest": "qk:def",
        "attn.q_rope_state": "post_rope",
        "attn.k_rope_state": "post_rope",
        "attn.k_cache_rope_state": "post_rope",
        "attn.fusion_boundary": "unfused_rope_attention",
    }
    result.update(overrides)
    return result


def test_contract_maps_actual_attention_state_onto_the_generic_schema():
    contract = adapter.build_contract(PolicyRole.ROLLOUT, _effective())

    assert contract.precision.compute is Precision.BF16
    assert contract.precision.accumulate is Precision.FP32
    assert contract.precision.softmax_accumulate is Precision.FP32
    assert contract.precision.downcast_at is DowncastPoint.FINAL_WRITE
    assert contract.collectives[0].op is CollectiveOp.POINT_TO_POINT
    assert contract.collectives[0].reduction_order is ReductionOrder.GLOBAL_BLOCK_INDEX
    assert len(contract.extra["split_kv_coordinates"]) == 4
    assert contract.extra["split_kv_boundaries"][0][1] == ((0, 2), (2, 4))
    assert contract.extra["cp_block_manifest"][1][1:3] == (4, 8)
    assert contract.extra["lse_domain"] == "attention"


def test_missing_actual_plan_is_not_filled_from_requested_policy():
    config = _effective()
    del config["attn.actual_split_kv_plan_set"]
    config["attn.requested_split_kv_policy"] = "fixed"
    config["attn.requested_split_kv_size"] = 2

    contract = adapter.build_contract(PolicyRole.TRAINING, config)
    assert contract.extra["requested_split_kv_policy"] == "fixed"
    assert "split_kv_runtime_plan_set" not in contract.extra
    assert "split_kv_boundaries" not in contract.extra


def test_incomplete_or_rank_variant_plan_set_fails_closed():
    missing = _plan_set()
    missing["entries"].pop()
    with pytest.raises(adapter.AttentionAdapterError, match="coverage is incomplete"):
        adapter.build_contract(
            PolicyRole.TRAINING,
            _effective(**{"attn.actual_split_kv_plan_set": missing}),
        )

    rank_variant = _plan_set()
    rank_variant["entries"][2]["actual_split_boundaries"] = [[0, 4]]
    with pytest.raises(adapter.AttentionAdapterError, match="differs across TP/CP consumers"):
        adapter.build_contract(
            PolicyRole.TRAINING,
            _effective(**{"attn.actual_split_kv_plan_set": rank_variant}),
        )


def test_reported_split_count_must_match_actual_boundaries():
    plan = _plan_set()
    plan["entries"][0]["actual_split_kv_count"] = 3
    with pytest.raises(adapter.AttentionAdapterError, match="must equal"):
        adapter.build_contract(
            PolicyRole.TRAINING,
            _effective(**{"attn.actual_split_kv_plan_set": plan}),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split_kv_accum_dtype", "bf16", "accumulate in fp32"),
        ("split_kv_downcast_at", "per_partial", "only at final_write"),
        ("split_kv_merge_order", "arrival", "global_block_index"),
    ],
)
def test_invalid_split_kv_numerical_contract_is_rejected(field, value, message):
    plan = _plan_set()
    plan["entries"][0][field] = value
    with pytest.raises(adapter.AttentionAdapterError, match=message):
        adapter.build_contract(
            PolicyRole.ROLLOUT,
            _effective(**{"attn.actual_split_kv_plan_set": plan}),
        )


def test_non_fp32_attention_accumulation_is_rejected_before_comparison():
    with pytest.raises(adapter.AttentionAdapterError, match="must accumulate in fp32"):
        adapter.build_contract(
            PolicyRole.TRAINING,
            _effective(**{"attn.accumulate_dtype": "bf16"}),
        )


def test_cp_manifest_must_be_complete_and_gap_free():
    manifest = _manifest()
    manifest[1]["kv_block_start"] = 5
    with pytest.raises(adapter.AttentionAdapterError, match="gap-free"):
        adapter.build_contract(
            PolicyRole.ROLLOUT,
            _effective(**{"attn.cp_block_manifest": manifest}),
        )


def test_reference_cp_switch_builds_the_p2p_contract_without_claiming_a_runtime_plan():
    config = {
        "attn.cp_world_size": 2,
        "attn.cp_merge": CP_MERGE_REFERENCE.name,
    }
    contract = adapter.build_contract(PolicyRole.TRAINING, config)
    assert contract.collectives[0].backend == "p2p_nccl_reference"
    assert contract.collectives[0].reduction_order is ReductionOrder.GLOBAL_BLOCK_INDEX
    assert "split_kv_runtime_plan_set" not in contract.extra


def test_role_specific_downcast_supports_a_training_only_ablation():
    config = _effective(**{"attn.training_downcast_at": "per_partial"})
    training = adapter.build_contract(PolicyRole.TRAINING, config)
    rollout = adapter.build_contract(PolicyRole.ROLLOUT, config)
    assert training.precision.downcast_at is DowncastPoint.PER_PARTIAL
    assert rollout.precision.downcast_at is DowncastPoint.FINAL_WRITE


def test_readback_accepts_mapping_reader_and_attribute_but_rejects_requested_only():
    assert adapter.read_effective_config(PolicyRole.TRAINING, _effective())["attn.batch_size"] == 1

    class Engine:
        role = PolicyRole.ROLLOUT

        def read_effective_config(self):
            return {"attn.cp_world_size": 2}

    assert adapter.read_effective_config(PolicyRole.ROLLOUT, Engine()) == {
        "attn.cp_world_size": 2
    }

    class Bare:
        effective_config = {"attn.compute_dtype": "bf16"}

    assert adapter.read_effective_config(PolicyRole.TRAINING, Bare()) == {
        "attn.compute_dtype": "bf16"
    }

    with pytest.raises(adapter.AttentionAdapterError, match="requested_config"):
        adapter.read_effective_config(
            PolicyRole.TRAINING, {"requested_config": {"attn.cp_world_size": 2}}
        )


def test_observed_collective_comes_from_effective_runtime_state():
    observed = adapter.observe_collectives(PolicyRole.ROLLOUT, _effective())
    assert observed[0].backend == "p2p_nccl_reference"

    config = _effective()
    del config["attn.cp_collective"]
    assert adapter.observe_collectives(PolicyRole.ROLLOUT, config) == ()


def test_implementation_resolution_has_a_trace_for_every_failed_candidate():
    impl, resolution = adapter.resolve_implementation(
        "attn.split_kv", PolicyRole.TRAINING, "does.not.exist.Missing"
    )
    assert impl is None
    assert resolution.resolved is None
    assert resolution.rejected

    impl, resolution = adapter.resolve_implementation(
        "attn.split_kv", PolicyRole.TRAINING, "math.sqrt"
    )
    assert impl(9.0) == 3.0
    assert resolution.resolved == "math.sqrt"


def test_plugin_wires_adapter_and_discovers_all_attention_factors():
    assert AttentionChecks.build_contract is adapter.build_contract
    ids = [factor.id for factor in AttentionChecks().declare_factors()]
    assert ids == [
        "attn.cp_merge",
        "attn.precision_downcast",
        "attn.rope_fusion",
        "attn.split_kv",
    ]


def test_plan_helper_is_not_mutated_by_contract_building():
    plan = _plan_set()
    before = deepcopy(plan)
    adapter.build_contract(
        PolicyRole.ROLLOUT,
        _effective(**{"attn.actual_split_kv_plan_set": plan}),
    )
    assert plan == before


def test_attention_extra_is_json_serializable_for_report_artifacts():
    contract = adapter.build_contract(PolicyRole.ROLLOUT, _effective())
    json.dumps(contract.extra)
