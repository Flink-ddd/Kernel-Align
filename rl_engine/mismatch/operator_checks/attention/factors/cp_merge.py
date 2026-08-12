# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""attn.cp_merge -- CP ownership, communication and (out, lse) merge order."""

from __future__ import annotations

from rl_engine.mismatch.operator_checks.attention._common import (
    ATTENTION_LSE_EVIDENCE,
    CP_BLOCK_MANIFEST_EVIDENCE,
    CP_MERGE_REFERENCE,
)
from rl_engine.mismatch.schema import (
    COLLECTIVE_CONTRACT,
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
    id="attn.cp_merge",
    operator="attention",
    category=FactorCategory.SHARDING_AND_REDUCTION,
    question=(
        "Does CP Attention drift because block ownership, communication, or the FP32 "
        "attention-domain (out, lse) merge order differs?"
    ),
    switch=Switch(
        path="attn.cp_merge",
        rebind_cost=RebindCost.PROCESS_GROUP_REBUILD,
        applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
        allowed_values=("native", CP_MERGE_REFERENCE.name),
    ),
    comparison_rules={
        "extra.tp_world_size": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.cp_world_size": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.cp_block_manifest": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.cp_owner_ranges": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.lse_domain": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.export_lse": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.merge_state": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].group": ComparisonRule.MUST_MATCH_BITWISE,
        "collectives[0].group_size": ComparisonRule.MUST_MATCH_BITWISE,
        "collectives[0].op": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].reduction_order": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].accumulate_precision": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].downcast_at": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].backend": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(
        required_ops=("p2p_nccl_attention_reference",),
        min_gpu_count=2,
    ),
    required_evidence=(
        Evidence.EFFECTIVE_CONFIG_READBACK.value,
        COLLECTIVE_CONTRACT,
        CP_BLOCK_MANIFEST_EVIDENCE,
        ATTENTION_LSE_EVIDENCE,
    ),
    reference=CP_MERGE_REFERENCE,
    pitfalls=(
        KnownPitfall(
            id="cp_arrival_order_merge",
            mode=FailureMode.SILENT_FALSE_NEGATIVE,
            symptom="the CP result is stable on one topology but moves on another",
            actual_cause=(
                "partial (out, lse) states were merged in arrival/NCCL order instead of "
                "logical global block order"
            ),
            guard=(
                "exchange an authoritative block manifest, sort by global_block_index, "
                "then merge (out, lse) in fp32"
            ),
            guard_runs_at=NoiseFloor.SHARDED_SINGLE_NODE,
        ),
    ),
)
