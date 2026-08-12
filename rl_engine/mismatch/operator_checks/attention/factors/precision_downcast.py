# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""attn.precision_downcast -- compute precision and final write boundary."""

from __future__ import annotations

from rl_engine.mismatch.schema import (
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
    id="attn.precision_downcast",
    operator="attention",
    category=FactorCategory.OUTPUT_NUMERICS,
    question=(
        "Does Attention drift because the compute dtype differs or an FP32 partial is "
        "downcast before the final output write?"
    ),
    switch=Switch(
        path="attn.training_downcast_at",
        rebind_cost=RebindCost.PER_REQUEST,
        applies_to=(PolicyRole.TRAINING,),
        allowed_values=("final_write", "per_partial", "per_block"),
    ),
    comparison_rules={
        "precision.compute": ComparisonRule.MUST_MATCH_BITWISE,
        "precision.accumulate": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "precision.softmax_accumulate": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "precision.downcast_at": ComparisonRule.MUST_MATCH_SEMANTICALLY,
    },
    prerequisites=Prerequisites(required_ops=("attention",)),
    required_evidence=(Evidence.EFFECTIVE_CONFIG_READBACK.value,),
    pitfalls=(
        KnownPitfall(
            id="partial_state_downcast_hidden",
            mode=FailureMode.SILENT_FALSE_NEGATIVE,
            symptom="the final output dtype matches but CP/Split-KV drift remains",
            actual_cause="one path rounded each partial before the online-softmax merge",
            guard="capture partial out/lse dtypes and require exactly one final-write downcast",
            guard_runs_at=NoiseFloor.SINGLE_LAYER_ANCHOR,
        ),
    ),
)
