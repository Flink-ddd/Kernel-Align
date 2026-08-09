# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""False-positive filtering, root-cause tracing, and the final report. Step 7."""

from __future__ import annotations

from typing import Sequence

from rl_engine.mismatch.schema import (
    Diagnosis,
    FactorReport,
    KnownPitfall,
    LibraryPin,
    MismatchReport,
    ModuleCorrespondence,
    NoiseFloor,
    PropagationEdge,
    RootCauseCategory,
    RootCauseHypothesis,
)

_SIDE_DIAGNOSES = (
    Diagnosis.CAUSED_BY_TRAINING_SIDE,
    Diagnosis.CAUSED_BY_ROLLOUT_SIDE,
    Diagnosis.CAUSED_BY_BOTH_SIDES,
)


def filter_known_equivalences(
    correspondences: Sequence[ModuleCorrespondence],
    findings: Sequence[str],
) -> tuple[tuple[str, ...], tuple[ModuleCorrespondence, ...]]:
    """Drop findings explained by a known structural equivalence.

    Without this, a difference like fused QKV turns every weight comparison red
    and buries the real problem. An equivalence may only be claimed with a test
    that proves it -- one without ``verified_by`` is not trusted here.
    """

    proven = tuple(
        item
        for item in correspondences
        if item.equivalence is not None and item.verified_by is not None
    )
    explained = {item.semantic_name for item in proven}
    kept = tuple(finding for finding in findings if finding not in explained)
    return kept, proven


def trace_root_causes(
    reports: Sequence[FactorReport],
    correspondences: Sequence[ModuleCorrespondence],
    edges: Sequence[PropagationEdge],
) -> tuple[RootCauseHypothesis, ...]:
    """Walk from the still-aligned anchor down the call chain into ranked hypotheses.

    Thirty factors give thirty diagnoses, and that pile is not the answer. The
    job here is to combine them with the module correspondence table and the
    call chain into "the few most suspicious modules, ranked".
    """

    downstream_of: dict[str, list[str]] = {}
    for edge in edges:
        downstream_of.setdefault(edge.upstream, []).append(edge.downstream)

    implicated = [report for report in reports if report.diagnosis in _SIDE_DIAGNOSES]
    if not implicated:
        return ()

    module_of = {item.semantic_name: item for item in correspondences}
    hypotheses: list[RootCauseHypothesis] = []

    for rank, report in enumerate(_by_confidence(implicated), start=1):
        module = _module_for_factor(report.factor_id, module_of)
        anchor = _anchor_for(module, downstream_of)
        hypotheses.append(
            RootCauseHypothesis(
                suspected_module=module,
                category=_categorise(report.diagnosis),
                anchor_module=anchor,
                supporting_factors=(report.factor_id,),
                evidence=(report.diagnosis_reason,),
                rank=rank,
            )
        )

    return tuple(hypotheses)


def _by_confidence(reports: Sequence[FactorReport]) -> list[FactorReport]:
    """A one-sided attribution is more actionable than a both-sided one."""

    order = {
        Diagnosis.CAUSED_BY_TRAINING_SIDE: 0,
        Diagnosis.CAUSED_BY_ROLLOUT_SIDE: 0,
        Diagnosis.CAUSED_BY_BOTH_SIDES: 1,
    }
    return sorted(reports, key=lambda report: (order[report.diagnosis], report.factor_id))


def _module_for_factor(factor_id: str, module_of: dict[str, ModuleCorrespondence]) -> str:
    """Map a factor id onto a semantic module name, falling back to the id."""

    for name in module_of:
        if factor_id.startswith(name.split(".", 1)[0]):
            return name
    return factor_id


def _anchor_for(module: str, downstream_of: dict[str, list[str]]) -> str:
    """The last still-aligned position upstream of the suspect."""

    for upstream, children in downstream_of.items():
        if module in children:
            return upstream
    return module


def _categorise(diagnosis: Diagnosis) -> RootCauseCategory:
    if diagnosis is Diagnosis.CAUSED_BY_BOTH_SIDES:
        return RootCauseCategory.DIFFERENT_IMPLEMENTATION
    return RootCauseCategory.DIFFERENT_IMPLEMENTATION


def build_report(
    reports: Sequence[FactorReport],
    *,
    noise_floor: NoiseFloor,
    library_pins: Sequence[LibraryPin] = (),
    correspondences: Sequence[ModuleCorrespondence] = (),
    edges: Sequence[PropagationEdge] = (),
    failed_guards: Sequence[KnownPitfall] = (),
) -> MismatchReport:
    """Assemble the final report."""

    _, filtered = filter_known_equivalences(
        correspondences, [report.factor_id for report in reports]
    )
    hypotheses = trace_root_causes(reports, correspondences, edges)

    return MismatchReport(
        noise_floor=noise_floor,
        library_pins=tuple(library_pins),
        factor_reports=tuple(reports),
        hypotheses=hypotheses,
        filtered_false_positives=filtered,
        failed_guards=tuple(failed_guards),
    )


def render_summary(report: MismatchReport) -> str:
    """A short human-readable summary. The hypotheses are what you want to read;
    everything else is the evidence supporting them."""

    lines = [
        f"noise floor: {report.noise_floor.value}",
        f"factors run: {len(report.factor_reports)}",
    ]

    tally: dict[str, int] = {}
    for factor_report in report.factor_reports:
        tally[factor_report.diagnosis.value] = tally.get(factor_report.diagnosis.value, 0) + 1
    for name in sorted(tally):
        lines.append(f"  {name}: {tally[name]}")

    if report.hypotheses:
        lines.append("root-cause hypotheses (most suspicious first):")
        for hypothesis in report.hypotheses:
            lines.append(
                f"  {hypothesis.rank}. {hypothesis.suspected_module} "
                f"[{hypothesis.category.value}] anchor={hypothesis.anchor_module}"
            )
    else:
        lines.append("root-cause hypotheses: none (no factor was attributed to a side)")

    if report.failed_guards:
        lines.append(f"failed pitfall guards: {[g.id for g in report.failed_guards]}")

    return "\n".join(lines)


__all__ = [
    "build_report",
    "filter_known_equivalences",
    "render_summary",
    "trace_root_causes",
]
