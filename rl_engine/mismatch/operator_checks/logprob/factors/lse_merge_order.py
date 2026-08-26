# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""logp.lse_merge_order -- how the per-shard LSE partials are merged under TP.

An implementation swap: training all_reduces (max, sumexp) partials in NCCL
order while rollout gathers full logits and reduces locally, and the reference
replaces both with WS2's fixed vocab-shard-order merge.
"""

from __future__ import annotations

from rl_engine.mismatch.operator_checks.logprob._common import DETERMINISTIC_LSE_REFERENCE
from rl_engine.mismatch.schema import (
    COLLECTIVE_CONTRACT,
    LSE_EXPORT,
    VOCAB_SHARD_MAP,
    ComparisonRule,
    Evidence,
    ExpectedOutcome,
    FactorCategory,
    FactorVariant,
    FailureMode,
    KnownPitfall,
    MismatchFactor,
    NoiseFloor,
    PolicyRole,
    Prerequisites,
    RebindCost,
    Switch,
)

_REF = DETERMINISTIC_LSE_REFERENCE

_VARIANTS = (
    FactorVariant(
        name="both_native",
        switch_values={"logp.lse_merge": "native"},
        why="baseline: each side merges its LSE partials the way its framework does",
    ),
    FactorVariant(
        name="both_reference",
        switch_values={"logp.lse_merge": _REF.name},
        replace_on={
            PolicyRole.ROLLOUT: _REF.rollout_impl,
            PolicyRole.TRAINING: _REF.training_impl,
        },
        expected=ExpectedOutcome.BITWISE_IDENTICAL,
        repeat_under={"NCCL_ALGO": ("Ring", "Tree"), "NCCL_PROTO": ("Simple", "LL")},
        why="self-check gate: a shard-order merge must survive a different NCCL algorithm",
    ),
    FactorVariant(
        name="training_reference_only",
        switch_values={"logp.lse_merge": f"{_REF.name}@training"},
        replace_on={PolicyRole.TRAINING: _REF.training_impl},
        why="swap the training side only: if the deviation goes, that side is the source",
    ),
    FactorVariant(
        name="rollout_reference_only",
        switch_values={"logp.lse_merge": f"{_REF.name}@rollout"},
        replace_on={PolicyRole.ROLLOUT: _REF.rollout_impl},
        why="swap the rollout side only: if the deviation goes, that side is the source",
    ),
)

FACTOR = MismatchFactor(
    id="logp.lse_merge_order",
    operator="logprob",
    category=FactorCategory.SHARDING_AND_REDUCTION,
    question=(
        "Does the logprob deviation come from the two sides merging their "
        "partial-LSE shards in different floating-point orders under TP?"
    ),
    switch=Switch(
        path="logp.lse_merge",
        rebind_cost=RebindCost.PROCESS_GROUP_REBUILD,
        applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
        allowed_values=("native", "rl_kernel"),
    ),
    comparison_rules={
        # Same tiers as gemm.forward_reduce for the shared collective paths;
        # the registry rejects one path declared at two different tiers.
        "collectives[0].op": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].reduction_order": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].accumulate_precision": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].downcast_at": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].group_size": ComparisonRule.MUST_MATCH_BITWISE,
        "collectives[0].backend": ComparisonRule.RECORD_ONLY,
        # Disagreeing shard maps make partial comparison meaningless: void, not
        # a finding.
        "extra.vocab_shard_map": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.lse_export": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(required_ops=("vocab_parallel_logp",), min_gpu_count=2),
    required_evidence=(
        Evidence.EFFECTIVE_CONFIG_READBACK.value,
        COLLECTIVE_CONTRACT,
        VOCAB_SHARD_MAP,
        LSE_EXPORT,
    ),
    reference=_REF,
    variants=_VARIANTS,
    pitfalls=(
        KnownPitfall(
            id="padded_vocab_in_lse",
            mode=FailureMode.SILENT_FALSE_NEGATIVE,
            symptom="a one-sided dlogp bias on every token, blamed on the merge order",
            actual_cause=(
                "padded vocab columns leak into one side's local sumexp, inflating "
                "its LSE denominator -- the merge order was never the problem"
            ),
            guard="assert exp(logp) sums to 1 over the real vocabulary on one anchor token",
            guard_runs_at=NoiseFloor.SINGLE_LAYER_ANCHOR,
        ),
    ),
)
