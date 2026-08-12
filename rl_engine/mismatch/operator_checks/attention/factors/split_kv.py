# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""attn.split_kv -- actual per-batch/TP/CP/owner Split-KV schedule."""

from __future__ import annotations

from rl_engine.mismatch.operator_checks.attention._common import (
    SPLIT_KV_PLAN_EVIDENCE,
    SPLIT_KV_REFERENCE,
)
from rl_engine.mismatch.schema import (
    BATCH_PLACEMENT,
    ComparisonRule,
    Evidence,
    FactorCategory,
    FailureMode,
    KnownPitfall,
    MismatchFactor,
    NoiseFloor,
    PolicyRole,
    Prerequisites,
    RebindCost,
    Switch,
)

FACTOR = MismatchFactor(
    id="attn.split_kv",
    operator="attention",
    category=FactorCategory.SHARDING_AND_REDUCTION,
    question=(
        "Does Attention drift because train and rollout executed different logical "
        "Split-KV boundaries or fallback schedules?"
    ),
    switch=Switch(
        path="attn.split_kv",
        rebind_cost=RebindCost.ENGINE_REBUILD,
        applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
        allowed_values=("native", SPLIT_KV_REFERENCE.name),
    ),
    comparison_rules={
        "extra.batch_size": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.tp_world_size": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.cp_world_size": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.total_kv_tokens": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.split_kv_coordinates": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.split_kv_owner_ranges": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.split_kv_boundaries": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.split_kv_merge_order": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.split_kv_accumulate_precision": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.split_kv_downcast_at": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.split_kv_fallback": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.split_kv_runtime_plan_set": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.requested_split_kv_policy": ComparisonRule.RECORD_ONLY,
        "extra.requested_split_kv_size": ComparisonRule.RECORD_ONLY,
        "extra.split_kv_backend": ComparisonRule.RECORD_ONLY,
        "extra.split_kv_plan_source": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(
        required_ops=("cp_attention",),
        min_gpu_count=2,
    ),
    required_evidence=(
        Evidence.EFFECTIVE_CONFIG_READBACK.value,
        BATCH_PLACEMENT,
        SPLIT_KV_PLAN_EVIDENCE,
    ),
    reference=SPLIT_KV_REFERENCE,
    pitfalls=(
        KnownPitfall(
            id="requested_split_kv_is_not_execution",
            mode=FailureMode.SILENT_FALSE_NEGATIVE,
            symptom="matching Split-KV policy scalars make the case look clean",
            actual_cause=(
                "runtime shape selection or fallback produced different actual boundaries "
                "on one batch/rank/owner"
            ),
            guard=(
                "require the complete batch x TP x CP x owner runtime plan set and compare "
                "actual boundaries"
            ),
            guard_runs_at=NoiseFloor.SHARDED_SINGLE_NODE,
        ),
    ),
)
