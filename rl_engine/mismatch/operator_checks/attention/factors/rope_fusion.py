# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""attn.rope_fusion -- RoPE fusion and tensor state."""

from __future__ import annotations

from rl_engine.mismatch.operator_checks.attention._common import (
    POST_ROPE_QK_EVIDENCE,
    TE_ROPE_REFERENCE,
)
from rl_engine.mismatch.schema import (
    POSITION_CACHE,
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
    id="attn.rope_fusion",
    operator="attention",
    category=FactorCategory.KERNEL_IMPLEMENTATION,
    question=(
        "Does the deviation come from fused vs small-operator vs sin/cos-cached "
        "RoPE, or from position_ids / theta / the cast boundary?"
    ),
    switch=Switch(
        path="attn.rope_fusion",
        rebind_cost=RebindCost.ENGINE_REBUILD,
        applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
        allowed_values=("native", "transformer_engine"),
    ),
    comparison_rules={
        "extra.rope_theta": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.position_ids_digest": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.post_rope_qk_digest": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.q_rope_state": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.k_rope_state": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.k_cache_rope_state": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "precision.downcast_at": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        # Fused vs unfused is what this factor ablates, so comparing it would
        # fail every arm by construction.
        "extra.fusion_boundary": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(
        required_ops=("rope",),
        required_packages=("transformer_engine>=2.0",),
    ),
    required_evidence=(
        Evidence.EFFECTIVE_CONFIG_READBACK.value,
        POSITION_CACHE,
        POST_ROPE_QK_EVIDENCE,
    ),
    reference=TE_ROPE_REFERENCE,
    pitfalls=(
        KnownPitfall(
            id="rope_hook_not_covered",
            mode=FailureMode.MISSING_INSTRUMENTATION,
            symptom="RoPE looks perfectly consistent between the two sides",
            actual_cause="the hook never attached, so nothing was captured at all",
            guard="dump post-RoPE Q/K on both sides and compare bitwise before ablating",
            guard_runs_at=NoiseFloor.SINGLE_LAYER_ANCHOR,
        ),
    ),
)
