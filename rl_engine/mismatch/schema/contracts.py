# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Numerical contracts and how the two sides are compared field by field."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from rl_engine.mismatch.schema.collectives import CollectiveContract
from rl_engine.mismatch.schema.values import PolicyRole, PrecisionProfile


class ComparisonRule(str, Enum):
    """What the framework does when this field differs between the two sides."""

    MUST_MATCH_BITWISE = "must_match_bitwise"  # differ -> this case is void
    MUST_MATCH_SEMANTICALLY = "must_match_semantically"  # implementations may differ
    RECORD_ONLY = "record_only"  # recorded, never compared


class ComparisonIssueCode(str, Enum):
    """Stable reason codes for the two sides disagreeing.

    These strings go into the artifact schema and callers branch on them, so
    renaming one requires a schema version bump.
    """

    REQUIRED_FIELD_MISSING = "required_field_missing"
    BITWISE_MISMATCH = "bitwise_mismatch"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    DETERMINISM_INCOMPATIBLE = "determinism_incompatible"


@dataclass(frozen=True)
class ComparisonIssue:
    """One record of the two sides disagreeing."""

    code: ComparisonIssueCode
    rule: ComparisonRule  # which tier this field was declared at
    field_path: str  # "collectives[0].reduction_order"
    values: Mapping[PolicyRole, Any]  # each side's actual value
    message: str = ""


@dataclass(frozen=True)
class OperatorContract:
    """One operator's numerical contract on one side.

    Only the three fields common to every operator live here; the rest goes in
    ``extra``, keyed by the paths a factor's ``comparison_rules`` declare::

        "precision.accumulate"
        "collectives[0].reduction_order"
        "extra.rope_theta"

    The framework indexes both contracts by path, so a plugin only has to put
    fields in the right place. Keep ``extra`` flat -- nesting makes paths long and
    unreadable.
    """

    operator: str
    role: PolicyRole
    precision: PrecisionProfile
    collectives: tuple[CollectiveContract, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ComparisonIssue",
    "ComparisonIssueCode",
    "ComparisonRule",
    "OperatorContract",
]
