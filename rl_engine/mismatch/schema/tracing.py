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
    """A training-side module paired with its rollout-side counterpart, plus any
    known structural difference between them.

    Without this table the two sides' tensors cannot even be matched up, let
    alone compared entry by entry.

    A non-empty ``equivalence`` means "different in form, equal in arithmetic" --
    the basis for filtering false positives. Without it, a difference like fused
    QKV turns every weight comparison red and buries the real problem.

    Filled once per model by whoever brings the model up, and shared by every
    operator: "Megatron's ``linear_fc1`` corresponds to vLLM's ``gate_up_proj``"
    is the same fact for attention, gemm and logprob alike. Swap in GLM5 or
    DSv3.2 and it has to be redone -- so it lives in ``model_meta/``, not in any
    ``operator_checks/``.
    """

    semantic_name: str  # "mlp.gate_up" -- framework-independent
    training_module: str  # "...megatron...linear_fc1"
    rollout_module: str  # "...vllm...gate_up_proj"
    equivalence: str | None = None  # "concat_on_dim0" / "transpose" / ...
    verified_by: str | None = None  # the test proving it. No proof, no claim.


@dataclass(frozen=True)
class PropagationEdge:
    """One directed edge of the call chain: mismatch at ``upstream`` propagates
    to ``downstream``.

    Tracing walks these backwards: start from the last still-aligned module and
    look downstream for the first mismatched one.
    """

    upstream: str  # semantic_name
    downstream: str


class RootCauseCategory(str, Enum):
    """How a root cause is classified."""

    MISSING_OPERATOR = "missing_operator"  # one side does not have it at all
    DIFFERENT_IMPLEMENTATION = "different_implementation"
    DIFFERENT_PARAMETER = "different_parameter"
    UPSTREAM_PROPAGATED = "upstream_propagated"  # fine itself, inherited from upstream


@dataclass(frozen=True)
class RootCauseHypothesis:
    """One root-cause hypothesis, produced by walking the call chain after a
    ``NOT_THIS_FACTOR``.

    ``anchor_module`` is the last position where the two sides still agree --
    the root cause must be downstream of it.
    """

    suspected_module: str
    category: RootCauseCategory
    anchor_module: str
    supporting_factors: tuple[str, ...]  # MismatchFactor.id
    evidence: tuple[str, ...]
    rank: int  # 1 is the most suspicious


@dataclass(frozen=True)
class MismatchReport:
    """The final report for one run. **Three levels away from a Diagnosis**::

        one factor's variants  ->  VariantResult x N
                                       | diagnose()
                                 FactorReport  (holds one Diagnosis enum value)
                                       | x 30 factors
                                 trace_root_causes()
                                       v
                                 MismatchReport  (ranked hypotheses)

    * ``Diagnosis`` is **one factor's** conclusion, one of eight values;
    * ``FactorReport`` is that factor's full record: every variant plus that one
      diagnosis plus the reason;
    * ``MismatchReport`` is the **whole run**, answering what a Diagnosis cannot.

    Thirty factors give thirty diagnoses -- say three ``CAUSED_BY_TRAINING_SIDE``,
    twenty-five ``NOT_THIS_FACTOR``, two ``INSUFFICIENT_EVIDENCE``. That pile is
    not the answer. **The report's job is to combine them with the module
    correspondence table and the call chain into "the few most suspicious
    modules, ranked"** -- the ``hypotheses`` field. The rest is the evidence
    supporting it.
    """

    noise_floor: NoiseFloor
    library_pins: tuple[LibraryPin, ...]
    factor_reports: tuple[FactorReport, ...]
    hypotheses: tuple[RootCauseHypothesis, ...]  # sorted by rank
    filtered_false_positives: tuple[ModuleCorrespondence, ...]  # those with equivalence set
    failed_guards: tuple[KnownPitfall, ...]


__all__ = [
    "MismatchReport",
    "ModuleCorrespondence",
    "PropagationEdge",
    "RootCauseCategory",
    "RootCauseHypothesis",
]
