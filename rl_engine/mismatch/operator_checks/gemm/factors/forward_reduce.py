# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""gemm.forward_reduce -- RowParallel forward reduction order."""

from __future__ import annotations

from rl_engine.mismatch.operator_checks.gemm._common import DETERMINISTIC_REDUCE_REFERENCE
from rl_engine.mismatch.schema import (
    COLLECTIVE_CONTRACT,
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

_REF = DETERMINISTIC_REDUCE_REFERENCE

# The standard four arms, spelled out so the self-check gate can also carry
# repeat_under: an implementation claiming topology independence must survive
# NCCL choosing a different algorithm.
_VARIANTS = (
    FactorVariant(
        name="both_native",
        switch_values={"gemm.forward_reduce": "native"},
        why="baseline: each side on its own framework's native reduction",
    ),
    FactorVariant(
        name="both_reference",
        switch_values={"gemm.forward_reduce": _REF.name},
        replace_on={
            PolicyRole.ROLLOUT: _REF.rollout_impl,
            PolicyRole.TRAINING: _REF.training_impl,
        },
        expected=ExpectedOutcome.BITWISE_IDENTICAL,
        repeat_under={"NCCL_ALGO": ("Ring", "Tree"), "NCCL_PROTO": ("Simple", "LL")},
        why="self-check gate, plus: a fixed order must survive a different NCCL algorithm",
    ),
    FactorVariant(
        name="training_reference_only",
        switch_values={"gemm.forward_reduce": f"{_REF.name}@training"},
        replace_on={PolicyRole.TRAINING: _REF.training_impl},
        why="swap the training side only: if the deviation goes, that side is the source",
    ),
    FactorVariant(
        name="rollout_reference_only",
        switch_values={"gemm.forward_reduce": f"{_REF.name}@rollout"},
        replace_on={PolicyRole.ROLLOUT: _REF.rollout_impl},
        why="swap the rollout side only: if the deviation goes, that side is the source",
    ),
)

FACTOR = MismatchFactor(
    id="gemm.forward_reduce",
    operator="gemm",
    category=FactorCategory.SHARDING_AND_REDUCTION,
    question=(
        "Does the RowParallel forward reduction differ because sequence "
        "parallelism rewrites all_reduce into reduce_scatter + all_gather?"
    ),
    switch=Switch(
        path="gemm.forward_reduce",
        rebind_cost=RebindCost.PROCESS_GROUP_REBUILD,
        applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
        allowed_values=("native", "rl_kernel"),
    ),
    comparison_rules={
        "collectives[0].op": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].reduction_order": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].accumulate_precision": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "collectives[0].group_size": ComparisonRule.MUST_MATCH_BITWISE,
        # Two backends may legitimately differ; what must agree is the order.
        "collectives[0].backend": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(required_ops=("ordered_reduce_scatter",), min_gpu_count=2),
    required_evidence=(Evidence.EFFECTIVE_CONFIG_READBACK.value, COLLECTIVE_CONTRACT),
    reference=_REF,
    # All three are row parallel linears eating the same accumulation order.
    call_sites=("attention.o_linear", "mlp.down_linear", "moe.output"),
    variants=_VARIANTS,
    pitfalls=(
        KnownPitfall(
            id="nccl_algo_unpinned",
            mode=FailureMode.SILENT_FALSE_NEGATIVE,
            symptom="the reduction-order conclusion looks stable",
            actual_cause="NCCL picks ring or tree per run, so the conclusion is noise",
            guard="pin NCCL_ALGO/NCCL_PROTO and rerun; results must be bitwise identical",
            guard_runs_at=NoiseFloor.SHARDED_SINGLE_NODE,
        ),
    ),
)
