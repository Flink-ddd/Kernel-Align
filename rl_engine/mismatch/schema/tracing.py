# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Module correspondence and root-cause tracing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rl_engine.mismatch.schema.metrics import FactorReport
from rl_engine.mismatch.schema.pitfalls import KnownPitfall
from rl_engine.mismatch.schema.values import LibraryPin
from rl_engine.mismatch.schema.variants import NoiseFloor


@dataclass(frozen=True)
class ModuleCorrespondence:
    """A training-side module paired with its rollout-side counterpart.

    A non-empty ``equivalence`` means "different in form, equal in arithmetic",
    which is what lets a false positive be filtered. Without it a difference like
    fused QKV turns every weight comparison red and buries the real problem.
    """

    semantic_name: str  # "mlp.gate_up" -- framework-independent
    training_module: str  # "...megatron...linear_fc1"
    rollout_module: str  # "...vllm...gate_up_proj"
    equivalence: str | None = None  # "concat_on_dim0" / "transpose" / ...
    verified_by: str | None = None  # the test proving it. No proof, no claim.


@dataclass(frozen=True)
class PropagationEdge:
    """One directed edge of the call chain, followed backwards when tracing."""

    upstream: str  # semantic_name
    downstream: str


class RootCauseCategory(str, Enum):
    MISSING_OPERATOR = "missing_operator"  # one side does not have it at all
    DIFFERENT_IMPLEMENTATION = "different_implementation"
    DIFFERENT_PARAMETER = "different_parameter"
    UPSTREAM_PROPAGATED = "upstream_propagated"  # fine itself, inherited from upstream


@dataclass(frozen=True)
class RootCauseHypothesis:
    """One hypothesis from walking the call chain after a ``NOT_THIS_FACTOR``.

    The root cause must be downstream of ``anchor_module``, the last position
    where the two sides still agree.
    """

    suspected_module: str
    category: RootCauseCategory
    anchor_module: str
    supporting_factors: tuple[str, ...]  # MismatchFactor.id
    evidence: tuple[str, ...]
    rank: int  # 1 is the most suspicious


@dataclass(frozen=True)
class MismatchReport:
    """The final report for one run.

    Thirty factors give thirty diagnoses, and that pile is not the answer. This
    combines them with the module correspondence table and the call chain into
    ``hypotheses`` -- the few most suspicious modules, ranked. The other fields
    are the evidence supporting it.
    """

    noise_floor: NoiseFloor
    library_pins: tuple[LibraryPin, ...]
    factor_reports: tuple[FactorReport, ...]
    hypotheses: tuple[RootCauseHypothesis, ...]  # sorted by rank
    filtered_false_positives: tuple[ModuleCorrespondence, ...]
    failed_guards: tuple[KnownPitfall, ...]


__all__ = [
    "MismatchReport",
    "ModuleCorrespondence",
    "PropagationEdge",
    "RootCauseCategory",
    "RootCauseHypothesis",
]
