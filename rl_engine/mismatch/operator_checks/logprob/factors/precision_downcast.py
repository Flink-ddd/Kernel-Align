# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""logp.precision_downcast -- lm_head dtype and where fp32 is written back."""

from __future__ import annotations

from rl_engine.mismatch.operator_checks.logprob._common import HEAD_DTYPES
from rl_engine.mismatch.schema import (
    MODEL_SHAPE,
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
    id="logp.precision_downcast",
    operator="logprob",
    category=FactorCategory.OUTPUT_NUMERICS,
    question=(
        "Is the deviation the lm_head GEMM running at the model dtype, or where "
        "the fp32 accumulator is written back?"
    ),
    # A parameter sweep, not an implementation swap: nothing is replaced, the
    # head dtype is scanned. ``reference=None`` is what says so -- there is no
    # separate "kind" field.
    switch=Switch(
        path="logp.head_dtype",
        # Cheapest tier: the dtype is a call argument, so arms of this factor
        # reuse one engine and sort ahead of every rebuild-level case.
        rebind_cost=RebindCost.PER_REQUEST,
        applies_to=(PolicyRole.TRAINING,),
        allowed_values=tuple(HEAD_DTYPES),
    ),
    comparison_rules={
        # The head GEMM's output dtype is identity, not a matter of taste.
        "precision.lm_head": ComparisonRule.MUST_MATCH_BITWISE,
        "precision.accumulate": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "precision.downcast_at": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        # Representation: both sides may name their mode differently.
        "extra.logprobs_mode": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(required_ops=("lm_head",)),
    required_evidence=(Evidence.EFFECTIVE_CONFIG_READBACK.value, MODEL_SHAPE),
    pitfalls=(
        KnownPitfall(
            id="head_dtype_tail_only",
            mode=FailureMode.SILENT_FALSE_NEGATIVE,
            symptom="dlogp_mean sits far below the clip edge, so the run reads as clean",
            actual_cause=(
                "a bf16 head is a rounding error on most tokens and a large one on "
                "the few with the flattest distribution -- the damage is all tail"
            ),
            guard="judge this factor on clip_fraction and dlogp_p99, never on the mean",
            guard_runs_at=NoiseFloor.SINGLE_LAYER_ANCHOR,
        ),
    ),
)
