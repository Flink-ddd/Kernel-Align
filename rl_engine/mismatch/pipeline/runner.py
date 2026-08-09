# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The execution loop: reuse decisions, variant repeats, metric computation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from rl_engine.mismatch.pipeline.comparison import compare_contracts
from rl_engine.mismatch.pipeline.registry import OperatorChecks
from rl_engine.mismatch.schema import (
    DEFAULT_CLIP_EPS,
    ComparisonIdentity,
    ExecutionPath,
    FactorVariant,
    MismatchFactor,
    MismatchMetrics,
    PolicyRole,
    ReuseKey,
    SwitchStatus,
    VariantResult,
    WorstToken,
)


class ScoringBackend(Protocol):
    """What the runner needs from an engine: logprobs for a fixed sequence.

    Minimal enough that a CPU stub can satisfy it, so the plumbing is testable
    without a GPU.
    """

    def score(
        self,
        role: PolicyRole,
        identity: ComparisonIdentity,
        switch_values: Mapping[str, Any],
        replacement: Callable[..., Any] | None,
    ) -> tuple[Sequence[float], Mapping[str, Any]]:
        """Return per-token logprobs plus whatever was read back."""
        ...

    def reuse_key(self, switch_values: Mapping[str, Any]) -> ReuseKey: ...


class ReadOnlyViolation(RuntimeError):
    """Computing logprobs modified the model."""


@dataclass(frozen=True)
class RunContext:
    """Everything one run needs that is not the factor itself."""

    identity: ComparisonIdentity
    path: ExecutionPath = ExecutionPath.TRAINING_FULL_PREFILL
    clip_eps: float = DEFAULT_CLIP_EPS
    strict: bool = True


def assert_comparison_is_read_only(before: str, after: str) -> None:
    """Model tensor fingerprints must match before and after scoring.

    A kernel that mutates weights in place makes ``both_reference`` fail bitwise
    at random, and the blame lands on the reference. This assertion is the
    self-check gate's own premise.
    """

    if before != after:
        raise ReadOnlyViolation(
            f"model state changed while computing logprobs: {before} -> {after}"
        )


def compute_metrics(
    rollout_logprobs: Sequence[float],
    training_logprobs: Sequence[float],
    active_mask: Sequence[bool],
    *,
    clip_eps: float = DEFAULT_CLIP_EPS,
    token_ids: Sequence[int] | None = None,
) -> MismatchMetrics:
    """Per-token metrics over active tokens only.

    ``dlogp = log pi_theta - log pi_old``, so ``rho = exp(dlogp)``, clipped at
    ``1 +/- eps``. Past that edge a token's gradient signal is discarded.
    """

    deltas: list[float] = []
    positions: list[int] = []
    for index, (roll, train, active) in enumerate(
        # strict: a length disagreement between logprobs and mask is a real bug
        zip(rollout_logprobs, training_logprobs, active_mask, strict=True)
    ):
        if not active:
            continue
        deltas.append(float(train) - float(roll))
        positions.append(index)

    if not deltas:
        return MismatchMetrics(
            active_token_count=0,
            dlogp_mean=0.0,
            dlogp_p99=0.0,
            dlogp_max=0.0,
            ratio_mean=1.0,
            ratio_max=1.0,
            clip_fraction=0.0,
            approx_kl=0.0,
        )

    magnitudes = [abs(delta) for delta in deltas]
    ratios = [math.exp(delta) for delta in deltas]
    upper_edge = math.log1p(clip_eps)
    lower_edge = -math.log1p(-clip_eps) if clip_eps < 1.0 else float("inf")
    clipped = sum(1 for delta in deltas if delta > upper_edge or delta < -abs(lower_edge))

    # k3 estimator: rho - 1 - ln(rho)
    approx_kl = sum(ratio - 1.0 - math.log(ratio) for ratio in ratios) / len(ratios)

    worst_index = max(range(len(deltas)), key=lambda i: magnitudes[i])
    worst = WorstToken(
        position=positions[worst_index],
        token_id=(token_ids[positions[worst_index]] if token_ids else -1),
        dlogp=deltas[worst_index],
    )

    return MismatchMetrics(
        active_token_count=len(deltas),
        dlogp_mean=sum(magnitudes) / len(magnitudes),
        dlogp_p99=_percentile(magnitudes, 0.99),
        dlogp_max=max(magnitudes),
        ratio_mean=sum(ratios) / len(ratios),
        ratio_max=max(ratios, key=lambda r: abs(math.log(r))),
        clip_fraction=clipped / len(deltas),
        approx_kl=approx_kl,
        worst_token=worst,
    )


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(math.ceil(q * len(ordered)) - 1))
    return ordered[max(0, index)]


def expand_repeats(variant: FactorVariant) -> tuple[Mapping[str, Any], ...]:
    """Expand ``repeat_under`` into the cartesian product of environments.

    The only exception to "one variant, one execution".
    """

    if not variant.repeat_under:
        return ({},)
    keys = sorted(variant.repeat_under)
    combos = itertools.product(*(variant.repeat_under[key] for key in keys))
    return tuple(dict(zip(keys, combo, strict=True)) for combo in combos)


def assert_order_is_topology_independent(results: Sequence[Sequence[float]]) -> bool:
    """Runs of a topology-independent collective must agree bitwise.

    Single-sided reruns, so the cheapest check in the whole set.
    """

    if len(results) < 2:
        return True
    first = list(results[0])
    return all(list(other) == first for other in results[1:])


def run_variant(
    factor: MismatchFactor,
    variant: FactorVariant,
    checks: OperatorChecks,
    backends: Mapping[PolicyRole, ScoringBackend],
    context: RunContext,
) -> VariantResult:
    """Run one variant on both sides and produce its result."""

    resolutions = {}
    replacements: dict[PolicyRole, Callable[..., Any] | None] = {}
    status = SwitchStatus.APPLIED

    for role in (PolicyRole.ROLLOUT, PolicyRole.TRAINING):
        wanted = (variant.replace_on or {}).get(role)
        if wanted is None:
            replacements[role] = None
            continue
        callable_impl, resolution = checks.resolve_implementation(factor.id, role, wanted)
        replacements[role] = callable_impl
        resolutions[role] = resolution
        if callable_impl is None:
            status = SwitchStatus.FELL_BACK

    scores: dict[PolicyRole, Sequence[float]] = {}
    readbacks: dict[PolicyRole, Mapping[str, Any]] = {}
    repeats: dict[PolicyRole, list[Sequence[float]]] = {
        PolicyRole.ROLLOUT: [],
        PolicyRole.TRAINING: [],
    }

    for environment in expand_repeats(variant):
        merged = {**variant.switch_values, **environment}
        for role in (PolicyRole.ROLLOUT, PolicyRole.TRAINING):
            logprobs, readback = backends[role].score(
                role, context.identity, merged, replacements[role]
            )
            repeats[role].append(logprobs)
            scores[role] = logprobs
            readbacks[role] = readback

    if variant.repeat_under:
        for role in (PolicyRole.ROLLOUT, PolicyRole.TRAINING):
            if not assert_order_is_topology_independent(repeats[role]):
                status = SwitchStatus.ERROR

    contracts = {
        role: checks.build_contract(role, variant.switch_values)
        for role in (PolicyRole.ROLLOUT, PolicyRole.TRAINING)
    }
    issues = compare_contracts(
        contracts[PolicyRole.ROLLOUT], contracts[PolicyRole.TRAINING], (factor,)
    )

    metrics = compute_metrics(
        scores[PolicyRole.ROLLOUT],
        scores[PolicyRole.TRAINING],
        context.identity.active_mask,
        clip_eps=context.clip_eps,
        token_ids=context.identity.response_token_ids,
    )

    evidence = frozenset(
        itertools.chain.from_iterable(readbacks[role].get("evidence", ()) for role in readbacks)
    )

    return VariantResult(
        variant=variant,
        path=context.path,
        status=status,
        metrics=metrics,
        evidence=evidence,
        effective_config={
            f"{role.value}.{key}": value
            for role, readback in readbacks.items()
            for key, value in readback.items()
            if key != "evidence"
        },
        collectives_observed=tuple(
            itertools.chain.from_iterable(contracts[role].collectives for role in contracts)
        ),
        resolution=resolutions.get(PolicyRole.TRAINING) or resolutions.get(PolicyRole.ROLLOUT),
        comparison_issues=issues,
    )


__all__ = [
    "ReadOnlyViolation",
    "RunContext",
    "ScoringBackend",
    "assert_comparison_is_read_only",
    "assert_order_is_topology_independent",
    "compute_metrics",
    "expand_repeats",
    "run_variant",
]
