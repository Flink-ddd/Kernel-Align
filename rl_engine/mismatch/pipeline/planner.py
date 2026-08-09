# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Planning: filter, statically reject, expand variants, order by rebuild cost.

Steps 2 to 4 of the pipeline.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Sequence

from rl_engine.mismatch.schema import (
    DeterminismLevel,
    ExpectedOutcome,
    FactorVariant,
    MismatchFactor,
    NoiseFloor,
    PolicyRole,
    RebindCost,
    ReductionOrder,
    declared_collectives,
    requires_fixed_order,
)

_NON_DETERMINISTIC_ORDERS = (ReductionOrder.NCCL_ALGORITHM, ReductionOrder.ARRIVAL)

_REBIND_ORDER = {
    RebindCost.PER_REQUEST: 0,
    RebindCost.ENGINE_REBUILD: 1,
    RebindCost.PROCESS_GROUP_REBUILD: 2,
    RebindCost.PROCESS_RESTART: 3,
}


class ContradictoryFactor(ValueError):
    """A factor's declaration contradicts itself; rejected before anything runs."""


@dataclass(frozen=True)
class UnmetPrerequisite:
    """One reason a factor cannot run today."""

    factor_id: str
    reason: str


def reject_contradictory_factors(factors: Sequence[MismatchFactor]) -> None:
    """Static check, run between steps 2 and 3. Nothing executes.

    Claiming topology independence while using a non-deterministic reduction
    order produces numbers that mean nothing. Reject at planning time rather
    than explaining it away after a full run.
    """

    for factor in factors:
        for contract in declared_collectives(factor):
            if requires_fixed_order(contract) and contract.reduction_order in (
                _NON_DETERMINISTIC_ORDERS
            ):
                raise ContradictoryFactor(
                    f"{factor.id}: claims {DeterminismLevel.STABLE_ACROSS_TOPOLOGY.value} "
                    f"but reduces with {contract.reduction_order.value}"
                )


def missing_prerequisites(
    factor: MismatchFactor,
    *,
    available_ops: frozenset[str] = frozenset(),
    gpu_count: int = 0,
    model_traits: frozenset[str] = frozenset(),
) -> tuple[UnmetPrerequisite, ...]:
    """What this factor is still missing: operators, devices, packages, model traits.

    A whitelist probe, not a hand-maintained blacklist of "unsupported" strings.
    """

    needs = factor.prerequisites
    unmet: list[UnmetPrerequisite] = []

    for op in needs.required_ops:
        if op not in available_ops:
            unmet.append(UnmetPrerequisite(factor.id, f"operator {op!r} is not dispatchable"))

    if gpu_count < needs.min_gpu_count:
        unmet.append(
            UnmetPrerequisite(
                factor.id, f"needs {needs.min_gpu_count} devices, {gpu_count} available"
            )
        )

    for requirement in needs.required_packages:
        package = requirement.split(">")[0].split("=")[0].split("<")[0].strip()
        if importlib.util.find_spec(package.replace("-", "_")) is None:
            unmet.append(UnmetPrerequisite(factor.id, f"package {requirement!r} is not installed"))

    for trait in needs.required_model_traits:
        if trait not in model_traits:
            unmet.append(UnmetPrerequisite(factor.id, f"model does not have trait {trait!r}"))

    for blocker in needs.blocked_by:
        unmet.append(UnmetPrerequisite(factor.id, f"blocked by {blocker}"))

    return tuple(unmet)


def build_variants(factor: MismatchFactor) -> tuple[FactorVariant, ...]:
    """Expand one factor into its variants.

    A single swap is not enough: **only a one-sided swap tells you which side is
    at fault, and only a two-sided swap proves the reference itself is sound.**
    Hence four arms rather than on/off.
    """

    if factor.variants:
        return tuple(factor.variants)

    path = factor.switch.path
    reference = factor.reference

    if reference is None:  # no reference implementation means a parameter sweep
        allowed = factor.switch.allowed_values or ()
        return tuple(
            FactorVariant(
                name=f"value_{value}",
                switch_values={path: value},
                why=f"sweep {path} = {value!r}",
            )
            for value in allowed
        )

    variants = [
        FactorVariant(
            name="both_native",
            switch_values={path: "native"},
            why="baseline: each side on its own framework's native implementation",
        ),
        FactorVariant(
            name="both_reference",
            switch_values={path: reference.name},
            replace_on={
                PolicyRole.ROLLOUT: reference.rollout_impl,
                PolicyRole.TRAINING: reference.training_impl,
            },
            expected=ExpectedOutcome.BITWISE_IDENTICAL,
            why=(
                "self-check gate: both sides on one implementation must agree bitwise, "
                "otherwise every conclusion from this factor is void"
            ),
        ),
        FactorVariant(
            name="training_reference_only",
            switch_values={path: f"{reference.name}@training"},
            replace_on={PolicyRole.TRAINING: reference.training_impl},
            why="swap the training side only: if the deviation goes, that side is the source",
        ),
        FactorVariant(
            name="rollout_reference_only",
            switch_values={path: f"{reference.name}@rollout"},
            replace_on={PolicyRole.ROLLOUT: reference.rollout_impl},
            why="swap the rollout side only: if the deviation goes, that side is the source",
        ),
    ]

    if reference.fp64_oracle is not None:
        variants.append(
            FactorVariant(
                name="fp64_oracle",
                switch_values={path: "fp64_oracle"},
                replace_on={
                    PolicyRole.ROLLOUT: reference.fp64_oracle,
                    PolicyRole.TRAINING: reference.fp64_oracle,
                },
                expected=ExpectedOutcome.BITWISE_IDENTICAL,
                why="gold standard: proves the reference used by both_reference is itself correct",
            )
        )

    return tuple(variants)


def order_cases_by_rebind_cost(
    cases: Sequence[tuple[MismatchFactor, FactorVariant]],
) -> tuple[tuple[MismatchFactor, FactorVariant], ...]:
    """Order cases so rebuild cost never decreases, maximising reuse.

    **Ordering is required, not a nicety**: run 160 cases in a random order and
    the worst case restarts the process for every one of them.
    """

    return tuple(sorted(cases, key=lambda item: _REBIND_ORDER[item[0].switch.rebind_cost]))


def suggested_floor_is_lowest(factor: MismatchFactor, floor: NoiseFloor) -> bool:
    """Whether this factor can even show anything at the given floor.

    A factor whose switch is process-level is identical on a single device, so
    running it at the anchor floor wastes machine time.
    """

    if factor.switch.rebind_cost is RebindCost.PROCESS_GROUP_REBUILD:
        return floor in (NoiseFloor.SHARDED_SINGLE_NODE, NoiseFloor.PRODUCTION)
    return True


__all__ = [
    "ContradictoryFactor",
    "UnmetPrerequisite",
    "build_variants",
    "missing_prerequisites",
    "order_cases_by_rebind_cost",
    "reject_contradictory_factors",
    "suggested_floor_is_lowest",
]
