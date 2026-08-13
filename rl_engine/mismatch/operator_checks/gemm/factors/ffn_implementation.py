# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""gemm.ffn_implementation -- Qwen3 FFN fast vs consistent arithmetic."""

from __future__ import annotations

from rl_engine.mismatch.operator_checks.gemm._common import (
    FFN_CONSISTENT_REFERENCE,
    FFN_STAGE_OUTPUTS,
)
from rl_engine.mismatch.schema import (
    MODEL_SHAPE,
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

_REF = FFN_CONSISTENT_REFERENCE

# Spelled out because the native baseline for PR2 is named ``fast`` rather than
# the framework-wide generic ``native`` value.
_VARIANTS = (
    FactorVariant(
        name="both_native",
        switch_values={"gemm.ffn_path": "fast"},
        why="baseline: both sides use the framework-native fast FFN path",
    ),
    FactorVariant(
        name="both_reference",
        switch_values={"gemm.ffn_path": _REF.name},
        replace_on={
            PolicyRole.ROLLOUT: _REF.rollout_impl,
            PolicyRole.TRAINING: _REF.training_impl,
        },
        expected=ExpectedOutcome.BITWISE_IDENTICAL,
        why="self-check gate: both sides use the same batch-invariant FFN path",
    ),
    FactorVariant(
        name="training_reference_only",
        switch_values={"gemm.ffn_path": f"{_REF.name}@training"},
        replace_on={PolicyRole.TRAINING: _REF.training_impl},
        why="replace only training FFN arithmetic to attribute a training-side mismatch",
    ),
    FactorVariant(
        name="rollout_reference_only",
        switch_values={"gemm.ffn_path": f"{_REF.name}@rollout"},
        replace_on={PolicyRole.ROLLOUT: _REF.rollout_impl},
        why="replace only rollout FFN arithmetic to attribute a rollout-side mismatch",
    ),
)

FACTOR = MismatchFactor(
    id="gemm.ffn_implementation",
    operator="gemm",
    category=FactorCategory.KERNEL_IMPLEMENTATION,
    question=(
        "Does Qwen3 FFN drift come from the native Gate/Up/SwiGLU/Down path "
        "rather than RL-Kernel's batch-invariant path?"
    ),
    switch=Switch(
        path="gemm.ffn_path",
        rebind_cost=RebindCost.PER_REQUEST,
        applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
        allowed_values=(
            "fast",
            _REF.name,
            f"{_REF.name}@training",
            f"{_REF.name}@rollout",
        ),
    ),
    comparison_rules={
        "precision.compute": ComparisonRule.MUST_MATCH_BITWISE,
        "precision.accumulate": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "precision.downcast_at": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.hidden_size": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.intermediate_size": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.tp_world_size": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.has_bias": ComparisonRule.MUST_MATCH_BITWISE,
        # These are the implementation choices under ablation. Comparing them
        # would reject the one-sided arms by construction.
        "extra.weight_layout": ComparisonRule.RECORD_ONLY,
        "extra.gate_up_packed": ComparisonRule.RECORD_ONLY,
        "extra.ffn_path": ComparisonRule.RECORD_ONLY,
        "extra.gemm_backend": ComparisonRule.RECORD_ONLY,
        "extra.activation_backend": ComparisonRule.RECORD_ONLY,
        "extra.batch_invariant": ComparisonRule.RECORD_ONLY,
        "extra.stage_output_digests": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(required_ops=("det_gemm", "swiglu")),
    required_evidence=(
        Evidence.EFFECTIVE_CONFIG_READBACK.value,
        MODEL_SHAPE,
        FFN_STAGE_OUTPUTS,
    ),
    reference=_REF,
    call_sites=("mlp.gate_up", "mlp.down"),
    variants=_VARIANTS,
    pitfalls=(
        KnownPitfall(
            id="ffn_requested_path_not_executed",
            mode=FailureMode.SILENT_FALSE_NEGATIVE,
            symptom="the one-sided replacement leaves the mismatch unchanged",
            actual_cause=(
                "the engine accepted the requested FFN path but silently executed "
                "its native GEMM or activation backend"
            ),
            guard=(
                "read module.provenance and capture Gate, Up, SwiGLU and Down "
                "stage digests for every arm"
            ),
            guard_runs_at=NoiseFloor.SINGLE_LAYER_ANCHOR,
        ),
    ),
)
