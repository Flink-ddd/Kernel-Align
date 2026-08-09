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

    STANDARD_FOUR = "standard_four"  # swap factors: both_native / both_reference / two one-sided
    VALUE_SWEEP = "value_sweep"  # sweep factors: one run per allowed value
    PAIRWISE = "pairwise"  # only after COUPLED_WITH_OTHER_FACTORS is diagnosed


class ExpectedOutcome(str, Enum):
    BITWISE_IDENTICAL = "bitwise_identical"  # failing means the reference is at fault
    MEASURE_ONLY = "measure_only"


class SwitchStatus(str, Enum):
    """Whether the switch actually reached the engine. A silent fallback is far
    more harmful than an error.

    Not ``MaterializationStatus`` -- it describes a **switch**'s state (did it
    get delivered), which "materialization" neither reveals nor keeps separate
    from attention's execution path.
    """

    APPLIED = "applied"
    FELL_BACK = "fell_back"  # requested, silently reverted to native
    UNSUPPORTED = "unsupported"
    UNOBSERVABLE = "unobservable"  # delivered but unreadable -- no evidence
    ERROR = "error"


class Diagnosis(str, Enum):
    """The conclusion for **one factor** after its variants have run -- an enum
    value, not a report.

    The first three are "cannot judge" and must stay strictly separate from
    "judged, nothing here". It answers "is this one factor the culprit"; it
    cannot answer "what causes the mismatch" -- that needs cross-factor
    synthesis, which is what ``MismatchReport`` is for.
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
    """The experiment's noise floor: how small a difference this run can resolve.
    Orthogonal to factors. A floor that has not passed blocks the next one.

    From signal processing: a signal below the noise floor cannot be measured.
    Same here -- at the production floor, an e-6 difference drowns in parallel
    reduction noise. Deliberately not named after scale: the levels are graded
    by noise, not by size.

    The four are arranged so that **each step down introduces exactly one new
    noise source**. When a floor starts failing, the suspect set is decided: it
    is whatever that floor just added.
    """

    SINGLE_LAYER_ANCHOR = "single_layer_anchor"
    # 1 layer / single device / determinism fully on / batch=1 / one token.
    # Noise sources: none, only the operator itself.
    # Called an anchor because it is the reference point for the other three:
    # **failing bitwise here is an operator bug, not "training-inference
    # mismatch"**, and the other three floors need not run at all.

    FULL_MODEL_SINGLE_GPU = "full_model_single_gpu"
    # All layers / still single device / determinism still on.
    # New: accumulation over depth. Tests whether error grows linearly or
    # exponentially with layer count.

    SHARDED_SINGLE_NODE = "sharded_single_node"
    # All layers / TP + SP on one node / determinism still on.
    # New: reduction-order differences from sharding.
    # **This is the first floor with "real" training-inference mismatch.**

    PRODUCTION = "production"
    # Target TP/CP/PP / determinism off / target batch / 2k-8k, decode path.
    # New: everything else (cross-node comms, non-deterministic kernels, the
    # real generation path). The floor reports are written from, and the only
    # one whose numbers may be read against the threshold table.


@dataclass(frozen=True)
class FactorVariant:
    """One arm of a controlled experiment -- the machine-readable form of
    "how do I configure this ablation"."""

    name: str
    switch_values: Mapping[str, Any]
    replace_on: Mapping[PolicyRole, str] | None = None
    expected: ExpectedOutcome = ExpectedOutcome.MEASURE_ONLY
    why: str = ""
    repeat_under: Mapping[str, tuple[Any, ...]] | None = None
    # Run this same variant once under each environment and require bitwise
    # equality -- **the only exception to "one variant, one execution"**. The
    # runner expands the cartesian product automatically.
    #
    #   repeat_under = {"NCCL_ALGO": ("Ring", "Tree"), "NCCL_PROTO": ("Simple", "LL")}
    #     -> four runs, asserted bitwise identical.
    #
    # It never compares across frameworks, so it is very cheap; and what it
    # verifies is the **premise of the self-check gate**: both_reference can
    # only serve as an anchor if the "fixed-order implementation" really did fix
    # the order. Without this, REFERENCE_ITSELF_IS_BROKEN is itself unreliable.


@dataclass(frozen=True)
class ExpectedRange:
    """The normal band for a metric under one (model family, noise floor, config).

    **Must be a code constant, never a config file** -- a tunable threshold means
    somebody can tune it until the test passes. It enters the execution
    fingerprint: changing a threshold requires a code change and review, and
    invalidates every historical pass/fail.
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
