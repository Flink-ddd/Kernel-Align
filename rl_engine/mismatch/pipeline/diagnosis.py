# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Four gates plus the diagnosis matrix. Step 6 of the pipeline.

The framework draws the conclusion; nobody reads the numbers by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rl_engine.mismatch.schema import (
    Diagnosis,
    ExpectedOutcome,
    FactorReport,
    KnownPitfall,
    MismatchFactor,
    MismatchMetrics,
    NoiseFloor,
    SwitchStatus,
    VariantResult,
    is_silent_failure,
    missing_evidence,
    tolerance_floor,
)

CONVERGENCE_RATIO = 0.1


@dataclass(frozen=True)
class _Outcome:
    diagnosis: Diagnosis
    reason: str


def diagnose(
    factor: MismatchFactor,
    variants: Sequence[VariantResult],
    *,
    noise_floor: NoiseFloor,
    model_family: str = "dense",
    failed_guards: Sequence[KnownPitfall] = (),
) -> FactorReport:
    """Run four gates, then the matrix.

    **"Not measured" and "measured and clean" are two different things** --
    confusing them is the mistake an attribution framework is most likely to
    make, so the gates come first and nothing reaches the matrix without passing
    them.
    """

    outcome = (
        _gate_variants_applied(variants)
        or _gate_evidence_complete(factor, variants)
        or _gate_shards_complete(variants)
        or _gate_guards_passed(failed_guards)
        or _run_matrix(variants, noise_floor=noise_floor, model_family=model_family)
    )

    return FactorReport(
        factor_id=factor.id,
        noise_floor=noise_floor,
        variants=tuple(variants),
        diagnosis=outcome.diagnosis,
        diagnosis_reason=outcome.reason,
    )


def _gate_variants_applied(variants: Sequence[VariantResult]) -> _Outcome | None:
    """Gate 1: did every variant actually take effect?

    ``FELL_BACK`` is the dangerous one -- the reference was requested, the engine
    silently reverted to native, and "the deviation did not change" then reads as
    ``NOT_THIS_FACTOR``. A false negative that looks exactly like a clean result.
    """

    for result in variants:
        if result.status is SwitchStatus.APPLIED:
            continue
        detail = ""
        if result.resolution is not None:
            rejected = ", ".join(
                f"{candidate.name} ({candidate.reason})" for candidate in result.resolution.rejected
            )
            detail = f"; tried: {rejected or 'nothing'}"
        return _Outcome(
            Diagnosis.VARIANT_DID_NOT_APPLY,
            f"variant {result.variant.name!r} is {result.status.value!r}{detail}",
        )
    return None


def _gate_evidence_complete(
    factor: MismatchFactor, variants: Sequence[VariantResult]
) -> _Outcome | None:
    """Gate 2: is the required evidence present?"""

    for result in variants:
        absent = missing_evidence(result.evidence, factor.required_evidence)
        if absent:
            return _Outcome(
                Diagnosis.INSUFFICIENT_EVIDENCE,
                f"variant {result.variant.name!r} is missing evidence: {sorted(absent)}",
            )
    return None


def _gate_shards_complete(variants: Sequence[VariantResult]) -> _Outcome | None:
    """Gate 3: were all logprob shards collected?

    Under TP/CP each rank holds one slice. Missing a slice makes the combined
    number wrong in a way that does not show: one vocab shard short and the LSE
    denominator loses a chunk, so logp comes out systematically high.
    """

    for result in variants:
        shards = result.logprob_shards
        if not shards:
            continue
        world_size = shards[0].world_size
        if len({shard.rank for shard in shards}) != world_size:
            return _Outcome(
                Diagnosis.INSUFFICIENT_EVIDENCE,
                f"variant {result.variant.name!r} collected {len(shards)} of "
                f"{world_size} logprob shards",
            )
    return None


def _gate_guards_passed(failed_guards: Sequence[KnownPitfall]) -> _Outcome | None:
    """Gate 4: did every pitfall guard pass?"""

    if failed_guards:
        names = ", ".join(guard.id for guard in failed_guards)
        return _Outcome(Diagnosis.INSUFFICIENT_EVIDENCE, f"pitfall guards did not pass: {names}")
    return None


def _run_matrix(
    variants: Sequence[VariantResult],
    *,
    noise_floor: NoiseFloor,
    model_family: str,
) -> _Outcome:
    """The matrix proper, once all four gates are clear."""

    by_name = {result.variant.name: result for result in variants}

    baseline = by_name.get("both_native")
    if baseline is None or baseline.metrics is None:
        return _Outcome(
            Diagnosis.INSUFFICIENT_EVIDENCE, "no both_native baseline to compare against"
        )

    self_check = by_name.get("both_reference")
    if self_check is not None:
        if not _is_bitwise_identical(self_check):
            return _Outcome(
                Diagnosis.REFERENCE_ITSELF_IS_BROKEN,
                "both_reference is not bitwise identical; fix the reference before "
                "trusting any conclusion from this factor",
            )

    training_only = by_name.get("training_reference_only")
    rollout_only = by_name.get("rollout_reference_only")
    if training_only is None or rollout_only is None:
        return _Outcome(
            Diagnosis.INSUFFICIENT_EVIDENCE,
            "one-sided variants are missing; a two-sided swap alone cannot attribute a side",
        )

    floor = tolerance_floor(model_family, noise_floor)
    training_converged = _converged(baseline.metrics, training_only.metrics, floor)
    rollout_converged = _converged(baseline.metrics, rollout_only.metrics, floor)

    if training_converged and not rollout_converged:
        return _Outcome(Diagnosis.CAUSED_BY_TRAINING_SIDE, "only the training-side swap converged")
    if rollout_converged and not training_converged:
        return _Outcome(Diagnosis.CAUSED_BY_ROLLOUT_SIDE, "only the rollout-side swap converged")
    if training_converged and rollout_converged:
        return _Outcome(
            Diagnosis.CAUSED_BY_BOTH_SIDES,
            "both one-sided swaps converged; the reference is the only anchor",
        )

    # Neither converged. Before calling it clean, check the tail: the mean stays
    # far below the clip edge at every production floor, so judging on the mean
    # alone would mark almost everything NOT_THIS_FACTOR.
    for candidate in (training_only, rollout_only):
        if candidate.metrics is not None and is_silent_failure(candidate.metrics):
            return _Outcome(
                Diagnosis.COUPLED_WITH_OTHER_FACTORS,
                "neither swap converged and the tail is past the clip edge; "
                "this factor interacts with another",
            )

    return _Outcome(Diagnosis.NOT_THIS_FACTOR, "neither one-sided swap moved the deviation")


def _is_bitwise_identical(result: VariantResult) -> bool:
    if result.variant.expected is not ExpectedOutcome.BITWISE_IDENTICAL:
        return True
    if result.metrics is None:
        return False
    return result.metrics.dlogp_max == 0.0


def _converged(
    baseline: MismatchMetrics, candidate: MismatchMetrics | None, tol_floor: float
) -> bool:
    """Convergence is judged on ``clip_fraction``, not ``dlogp_mean``.

    At the production floor the mean is always far below the clip edge, so using
    it would call nearly every factor ``NOT_THIS_FACTOR``.
    """

    if candidate is None:
        return False
    if baseline.clip_fraction > 0.0:
        return candidate.clip_fraction <= CONVERGENCE_RATIO * baseline.clip_fraction
    return candidate.dlogp_mean <= max(CONVERGENCE_RATIO * baseline.dlogp_mean, tol_floor)


__all__ = ["CONVERGENCE_RATIO", "diagnose"]
