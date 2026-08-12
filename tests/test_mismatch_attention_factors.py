# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Attention factor declarations and contract-comparison behavior."""

from __future__ import annotations

from copy import deepcopy

from rl_engine.mismatch.operator_checks.attention import adapter
from rl_engine.mismatch.operator_checks.attention.factors.cp_merge import FACTOR as CP_FACTOR
from rl_engine.mismatch.operator_checks.attention.factors.precision_downcast import (
    FACTOR as PRECISION_FACTOR,
)
from rl_engine.mismatch.operator_checks.attention.factors.rope_fusion import FACTOR as ROPE_FACTOR
from rl_engine.mismatch.operator_checks.attention.factors.split_kv import FACTOR as SPLIT_FACTOR
from rl_engine.mismatch.pipeline import (
    build_variants,
    compare_contracts,
    reject_contradictory_factors,
)
from rl_engine.mismatch.schema import ComparisonIssueCode, PolicyRole
from tests.test_mismatch_attention_adapter import _effective, _manifest, _plan_set


def _compare(config_rollout: dict, config_training: dict, factor):
    return compare_contracts(
        adapter.build_contract(PolicyRole.ROLLOUT, config_rollout),
        adapter.build_contract(PolicyRole.TRAINING, config_training),
        (factor,),
    )


def test_equal_requested_policy_but_different_actual_boundaries_is_a_finding():
    rollout = _effective()
    training = _effective()
    training["attn.actual_split_kv_plan_set"] = _plan_set(
        boundaries={0: [[0, 4]], 1: [[4, 8]]}
    )

    issues = _compare(rollout, training, SPLIT_FACTOR)
    paths = {issue.field_path for issue in issues}
    assert "extra.split_kv_boundaries" in paths
    assert "extra.split_kv_runtime_plan_set" in paths
    assert all(
        issue.code is ComparisonIssueCode.SEMANTIC_MISMATCH
        for issue in issues
        if issue.field_path in paths
    )


def test_missing_runtime_plan_is_reported_as_missing_not_clean():
    rollout = _effective()
    training = _effective()
    del training["attn.actual_split_kv_plan_set"]

    issues = _compare(rollout, training, SPLIT_FACTOR)
    missing = {
        issue.field_path
        for issue in issues
        if issue.code is ComparisonIssueCode.REQUIRED_FIELD_MISSING
    }
    assert "extra.split_kv_runtime_plan_set" in missing
    assert "extra.split_kv_boundaries" in missing


def test_split_kv_backend_and_trace_source_are_provenance_only():
    rollout = _effective()
    training = _effective()
    plan = training["attn.actual_split_kv_plan_set"]
    for entry in plan["entries"]:
        entry["split_kv_backend"] = "different-but-equivalent-backend"
        entry["split_kv_plan_source"] = "different-runtime-hook"

    assert _compare(rollout, training, SPLIT_FACTOR) == ()


def test_merge_order_and_collective_path_mismatches_are_visible():
    rollout = _effective()
    training = _effective()
    training["attn.cp_collective"] = {
        **training["attn.cp_collective"],
        "op": "all_gather",
        "reduction_order": "nccl_algorithm",
        "determinism": "none",
        "backend": "nccl",
    }

    issues = _compare(rollout, training, CP_FACTOR)
    paths = {issue.field_path for issue in issues}
    assert "collectives[0].op" in paths
    assert "collectives[0].reduction_order" in paths
    assert "collectives[0].backend" not in paths


def test_cp_owner_manifest_mismatch_voids_the_comparison_identity():
    rollout = _effective()
    training = _effective()
    manifest = deepcopy(_manifest())
    manifest[0]["owner_cp_rank"] = 1
    manifest[1]["owner_cp_rank"] = 0
    training["attn.cp_block_manifest"] = manifest

    issues = _compare(rollout, training, CP_FACTOR)
    by_path = {issue.field_path: issue for issue in issues}
    assert by_path["extra.cp_block_manifest"].code is ComparisonIssueCode.BITWISE_MISMATCH


def test_downcast_and_compute_dtype_mismatch_are_separate_findings():
    rollout = _effective()
    training = _effective(
        **{
            "attn.compute_dtype": "fp16",
            "attn.training_downcast_at": "per_partial",
        }
    )
    issues = _compare(rollout, training, PRECISION_FACTOR)
    by_path = {issue.field_path: issue for issue in issues}
    assert by_path["precision.compute"].code is ComparisonIssueCode.BITWISE_MISMATCH
    assert by_path["precision.downcast_at"].code is ComparisonIssueCode.SEMANTIC_MISMATCH


def test_rope_position_theta_and_post_qk_state_are_compared():
    rollout = _effective()
    training = _effective(
        **{
            "attn.rope_theta": 10_000.0,
            "attn.position_ids_digest": "positions:other",
            "attn.post_rope_qk_digest": "qk:other",
        }
    )
    issues = _compare(rollout, training, ROPE_FACTOR)
    assert {issue.field_path for issue in issues} >= {
        "extra.rope_theta",
        "extra.position_ids_digest",
        "extra.post_rope_qk_digest",
    }


def test_reference_factors_expand_to_four_arms_and_static_checks_pass():
    reject_contradictory_factors((SPLIT_FACTOR, CP_FACTOR, ROPE_FACTOR, PRECISION_FACTOR))
    for factor in (SPLIT_FACTOR, CP_FACTOR, ROPE_FACTOR):
        assert [variant.name for variant in build_variants(factor)] == [
            "both_native",
            "both_reference",
            "training_reference_only",
            "rollout_reference_only",
        ]
    assert [variant.name for variant in build_variants(PRECISION_FACTOR)] == [
        "value_final_write",
        "value_per_partial",
        "value_per_block",
    ]
