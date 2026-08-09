# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Contract comparison, driven entirely by declarations.

The comparison loop is generic; the per-operator part is putting fields in the
right place inside ``build_contract()``. There are no operator branches here.
"""

from __future__ import annotations

import math
import re
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Sequence

from rl_engine.mismatch.schema import (
    ComparisonIssue,
    ComparisonIssueCode,
    ComparisonRule,
    DeterminismLevel,
    MismatchFactor,
    OperatorContract,
    PolicyRole,
)

_MISSING = object()
_INDEX = re.compile(r"^(?P<name>[^\[]+)\[(?P<index>\d+)\]$")


def resolve_field_path(contract: OperatorContract, path: str) -> Any:
    """Index a contract by a dotted path such as ``collectives[0].reduction_order``.

    Returns a sentinel when the path is absent so a missing field is reported
    rather than raising.
    """

    current: Any = contract
    for part in path.split("."):
        matched = _INDEX.match(part)
        index = None
        if matched:
            part = matched.group("name")
            index = int(matched.group("index"))

        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
        elif is_dataclass(current) and any(f.name == part for f in fields(current)):
            current = getattr(current, part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return _MISSING

        if index is not None:
            if not isinstance(current, Sequence) or index >= len(current):
                return _MISSING
            current = current[index]

    return current


def _values_equal(left: Any, right: Any, *, bitwise: bool) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
        return left == right if bitwise else math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)
    return left == right


def compare_contracts(
    rollout: OperatorContract,
    training: OperatorContract,
    factors: Sequence[MismatchFactor],
) -> tuple[ComparisonIssue, ...]:
    """Compare the two contracts field by field, per ``comparison_rules``.

    Returns every disagreement. ``RECORD_ONLY`` fields are never compared -- that
    tier exists precisely so structural differences (packed QKV and the like) do
    not drown the real problems in false positives.
    """

    rules: dict[str, ComparisonRule] = {}
    for factor in factors:
        rules.update(factor.comparison_rules)

    issues: list[ComparisonIssue] = []
    for path, rule in sorted(rules.items()):
        if rule is ComparisonRule.RECORD_ONLY:
            continue

        left = resolve_field_path(rollout, path)
        right = resolve_field_path(training, path)

        if left is _MISSING or right is _MISSING:
            issues.append(
                ComparisonIssue(
                    code=ComparisonIssueCode.REQUIRED_FIELD_MISSING,
                    rule=rule,
                    field_path=path,
                    values={
                        PolicyRole.ROLLOUT: None if left is _MISSING else left,
                        PolicyRole.TRAINING: None if right is _MISSING else right,
                    },
                    message=(
                        f"{path!r} is declared {rule.value!r} but is absent from "
                        f"{'rollout' if left is _MISSING else 'training'}'s contract"
                    ),
                )
            )
            continue

        bitwise = rule is ComparisonRule.MUST_MATCH_BITWISE
        if _values_equal(left, right, bitwise=bitwise):
            continue

        code = (
            ComparisonIssueCode.BITWISE_MISMATCH
            if bitwise
            else ComparisonIssueCode.SEMANTIC_MISMATCH
        )
        issues.append(
            ComparisonIssue(
                code=code,
                rule=rule,
                field_path=path,
                values={PolicyRole.ROLLOUT: left, PolicyRole.TRAINING: right},
                message=f"{path!r} differs: rollout={left!r} training={right!r}",
            )
        )

    issues.extend(_determinism_issues(rollout, training))
    return tuple(issues)


def _determinism_issues(
    rollout: OperatorContract, training: OperatorContract
) -> tuple[ComparisonIssue, ...]:
    """One side claiming a stronger reproducibility guarantee than the other.

    Comparing a topology-independent implementation against one that is not
    reproducible even across runs measures the weaker side's noise, not the gap
    between them.
    """

    strength = {
        DeterminismLevel.NONE: 0,
        DeterminismLevel.STABLE_WITHIN_PROCESS: 1,
        DeterminismLevel.STABLE_ACROSS_RUNS: 2,
        DeterminismLevel.STABLE_ACROSS_TOPOLOGY: 3,
    }
    issues: list[ComparisonIssue] = []
    # strict=False: the two sides may legitimately declare different numbers of
    # collectives; the overlap is what can be compared here.
    paired = zip(rollout.collectives, training.collectives, strict=False)
    for index, (left, right) in enumerate(paired):
        if strength[left.determinism] != strength[right.determinism]:
            issues.append(
                ComparisonIssue(
                    code=ComparisonIssueCode.DETERMINISM_INCOMPATIBLE,
                    rule=ComparisonRule.MUST_MATCH_SEMANTICALLY,
                    field_path=f"collectives[{index}].determinism",
                    values={
                        PolicyRole.ROLLOUT: left.determinism,
                        PolicyRole.TRAINING: right.determinism,
                    },
                    message=(
                        f"collectives[{index}]: rollout guarantees "
                        f"{left.determinism.value!r} while training guarantees "
                        f"{right.determinism.value!r}"
                    ),
                )
            )
    return tuple(issues)


__all__ = ["compare_contracts", "resolve_field_path"]
