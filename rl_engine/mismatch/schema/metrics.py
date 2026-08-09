# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Metrics and per-variant results.

``dlogp`` is an intermediate quantity. GRPO's gradient enters the objective
through the importance sampling ratio::

    rho_{i,t}(theta) = pi_theta(y | x) / pi_old(y | x) = exp(dlogp_{i,t})

The objective clips rho, typically at eps = 0.2, i.e. rho in [0.8, 1.2]. Back in
dlogp terms, anything past ``ln(1 + eps) ~= 0.182`` **gets clipped**.

That is the actual mechanism by which mismatch breaks training: it raises the
clip fraction, and clipped tokens have their gradient signal cut -- **RL discards
samples it should have learned from**, and not random ones, the ones with the
largest mismatch.

Against the threshold table: dense sits at ``dlogp_mean ~= 0.002-0.008`` and
large MoE at ``0.01-0.03`` -- all far below 0.182. **So the mean never triggers
clipping, and looking only at the mean necessarily concludes "everything is
fine".** The danger is entirely in the tail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rl_engine.mismatch.schema.collectives import CollectiveContract
from rl_engine.mismatch.schema.contracts import ComparisonIssue
from rl_engine.mismatch.schema.values import ExecutionPath
from rl_engine.mismatch.schema.variants import (
    Diagnosis,
    FactorVariant,
    NoiseFloor,
    SwitchStatus,
)

DEFAULT_CLIP_EPS = 0.2


@dataclass(frozen=True)
class WorstToken:
    """The largest-deviation token -- the entry point for attribution, since it
    often points straight at a layer or an expert."""

    position: int
    token_id: int
    dlogp: float
    layer_hint: str | None = None
    expert_hint: int | None = None


@dataclass(frozen=True)
class MismatchMetrics:
    """Every metric from one comparison. Active tokens only."""

    active_token_count: int
    dlogp_mean: float
    dlogp_p99: float
    dlogp_max: float
    ratio_mean: float  # rho = exp(dlogp)
    ratio_max: float
    clip_fraction: float  # share of active tokens past the clip edge
    approx_kl: float  # k3 estimator: rho - 1 - ln(rho)
    worst_token: WorstToken | None = None


@dataclass(frozen=True)
class RejectedCandidate:
    """A rejected candidate implementation and why it was rejected."""

    name: str
    reason: str


@dataclass(frozen=True)
class ImplementationResolution:
    """Which candidates were tried and why each was rejected -- what you look at
    when the status is ``FELL_BACK``.

    A single ``fallback_reason: str`` is not enough: what you actually want is
    "was the first choice rejected for a missing library, missing devices, or a
    version mismatch", plus "which candidate did it land on". Without this trace,
    a silent fallback leaves nothing to go on.
    """

    requested: str
    resolved: str | None  # None = no candidate was usable
    rejected: tuple[RejectedCandidate, ...] = ()  # in the order tried


@dataclass(frozen=True)
class LogprobShard:
    """The slice of logprobs one rank holds under TP/CP.

    With TP/CP on, logprobs are not computed by any single rank -- each holds one
    slice. **Missing any slice makes the combined result wrong in a way that does
    not show**: drop one vocab shard and the LSE denominator loses a chunk, so
    logp is systematically too high. Slices must be collected per rank and
    checked against ``world_size`` before merging.

    Not ``RankObservation`` -- in RL, an observation is what the agent sees of
    the environment.
    """

    rank: int
    world_size: int
    selected_logprobs: Any  # torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantResult:
    """What one variant produced."""

    variant: FactorVariant
    path: ExecutionPath
    status: SwitchStatus
    metrics: MismatchMetrics | None
    evidence: frozenset[str]  # Evidence values plus plugin constants
    effective_config: Mapping[str, Any]  # read back, not requested
    collectives_observed: tuple[CollectiveContract, ...] = ()
    resolution: ImplementationResolution | None = None  # required unless APPLIED
    comparison_issues: tuple[ComparisonIssue, ...] = ()
    logprob_shards: tuple[LogprobShard, ...] = ()


@dataclass(frozen=True)
class FactorReport:
    """The conclusion for one factor, from all of its variants together."""

    factor_id: str
    noise_floor: NoiseFloor
    variants: tuple[VariantResult, ...]
    diagnosis: Diagnosis
    diagnosis_reason: str


def is_silent_failure(metrics: MismatchMetrics) -> bool:
    """Mean within band but the tail already past the clip edge.

    The most common false negative: the headline number looks healthy while
    gradient signal is being discarded.
    """

    return metrics.dlogp_mean < 0.01 and metrics.clip_fraction > 0.0


def missing_evidence(collected: frozenset[str], required: tuple[str, ...]) -> frozenset[str]:
    """Which required evidence is absent.

    A free function rather than a method on a bundle type: making a data
    structure reach into a factor's fields would be feature envy.
    """

    return frozenset(required) - collected


__all__ = [
    "DEFAULT_CLIP_EPS",
    "FactorReport",
    "ImplementationResolution",
    "LogprobShard",
    "MismatchMetrics",
    "RejectedCandidate",
    "VariantResult",
    "WorstToken",
    "is_silent_failure",
    "missing_evidence",
]
