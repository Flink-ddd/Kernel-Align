# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Variants, diagnosis, and the noise floor ladder."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from rl_engine.mismatch.schema.values import PolicyRole


class VariantExpansion(str, Enum):
    """Which variants a factor expands into."""

    STANDARD_FOUR = "standard_four"  # swap factors
    VALUE_SWEEP = "value_sweep"  # sweep factors: one run per allowed value
    PAIRWISE = "pairwise"  # only after COUPLED_WITH_OTHER_FACTORS is diagnosed


class ExpectedOutcome(str, Enum):
    BITWISE_IDENTICAL = "bitwise_identical"  # failing means the reference is at fault
    MEASURE_ONLY = "measure_only"


class SwitchStatus(str, Enum):
    """Whether the switch actually reached the engine.

    A silent fallback is far more harmful than an error.
    """

    APPLIED = "applied"
    FELL_BACK = "fell_back"  # requested, silently reverted to native
    UNSUPPORTED = "unsupported"
    UNOBSERVABLE = "unobservable"  # delivered but unreadable -- no evidence
    ERROR = "error"


class Diagnosis(str, Enum):
    """One factor's conclusion after its variants have run.

    The first three mean "cannot judge" and must stay strictly separate from
    "judged, nothing here". Answering what causes the mismatch overall needs
    cross-factor synthesis, which is ``MismatchReport``.
    """

    VARIANT_DID_NOT_APPLY = "variant_did_not_apply"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFERENCE_ITSELF_IS_BROKEN = "reference_itself_is_broken"
    CAUSED_BY_TRAINING_SIDE = "caused_by_training_side"
    CAUSED_BY_ROLLOUT_SIDE = "caused_by_rollout_side"
    CAUSED_BY_BOTH_SIDES = "caused_by_both_sides"
    NOT_THIS_FACTOR = "not_this_factor"
    COUPLED_WITH_OTHER_FACTORS = "coupled_with_other_factors"


class NoiseFloor(str, Enum):
    """How small a difference this run can resolve. Orthogonal to factors.

    Each step down introduces exactly one new noise source, so when a floor
    starts failing the suspect set is whatever that floor just added. A floor
    that has not passed blocks the next one.
    """

    SINGLE_LAYER_ANCHOR = "single_layer_anchor"
    # 1 layer, single device, determinism on, one token. No noise sources at all,
    # which is why failing bitwise here is an operator bug rather than mismatch,
    # and the other three floors need not run.

    FULL_MODEL_SINGLE_GPU = "full_model_single_gpu"
    # All layers, still single device. New: accumulation over depth. Tests
    # whether error grows linearly or exponentially with layer count.

    SHARDED_SINGLE_NODE = "sharded_single_node"
    # TP + SP on one node. New: reduction-order differences from sharding, so
    # this is the first floor with real training-inference mismatch.

    PRODUCTION = "production"
    # Target TP/CP/PP, determinism off, decode path. New: everything else. The
    # only floor whose numbers may be read against the threshold table.


@dataclass(frozen=True)
class FactorVariant:
    """One arm of a controlled experiment, as pasteable switch values.

    ``repeat_under`` runs this same arm once per environment and requires bitwise
    equality -- the only exception to "one variant, one execution", expanded by
    the runner as a cartesian product. It never compares across frameworks, so it
    is cheap, and it verifies the premise of the self-check gate: an arm can only
    anchor the others if its fixed-order implementation really did fix the order.
    """

    name: str
    switch_values: Mapping[str, Any]
    replace_on: Mapping[PolicyRole, str] | None = None
    expected: ExpectedOutcome = ExpectedOutcome.MEASURE_ONLY
    why: str = ""
    repeat_under: Mapping[str, tuple[Any, ...]] | None = None


@dataclass(frozen=True)
class ExpectedRange:
    """The normal band for a metric under one (model family, noise floor, config).

    A code constant, never configuration: a tunable threshold is one somebody
    tunes until the test passes. It enters the execution fingerprint, so changing
    it invalidates every historical pass/fail.
    """

    model_family: str  # "dense" / "moe" / "large_moe" / "*"
    noise_floor: NoiseFloor
    routing_replay: bool | None  # None = not applicable
    dlogp_mean: tuple[float, float]  # normal band [low, high]
    suspect_above: float
    note: str = ""


__all__ = [
    "Diagnosis",
    "ExpectedOutcome",
    "ExpectedRange",
    "FactorVariant",
    "NoiseFloor",
    "SwitchStatus",
    "VariantExpansion",
]
