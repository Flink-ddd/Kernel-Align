# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Known pitfalls, encoded as data.

Every factor in the docs carries a "pitfall" note. Prose pitfalls get read once
and never again -- they have to be data, so the framework can block them before
a run instead of relying on somebody remembering afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rl_engine.mismatch.schema.variants import NoiseFloor


class FailureMode(str, Enum):
    """How a pitfall fails -- decides which tool the framework needs against it.

    Borrowed from reliability engineering's failure mode, which is more accurate
    than a bare "kind": the useful part is *how it fools you*, not which bucket
    it sits in.
    """

    STRUCTURAL_FALSE_POSITIVE = "structural_false_positive"  # differs in form, equal in math
    SILENT_FALSE_NEGATIVE = "silent_false_negative"  # metrics look fine, conclusion is wrong
    MISSING_INSTRUMENTATION = "missing_instrumentation"  # never captured in the first place
    CONFIG_DEFAULT_TRAP = "config_default_trap"  # the default is not what you assumed
    CONVENTION_MISMATCH = "convention_mismatch"  # shift-by-one, log base, ...
    RESOURCE_LIMIT = "resource_limit"  # this arm simply cannot run


@dataclass(frozen=True)
class KnownPitfall:
    """A known pitfall together with the assertion that blocks it.

    ``symptom`` and ``actual_cause`` are separate on purpose: a pitfall is a
    pitfall precisely because its appearance points at the wrong cause.

    Originally split into ``Pitfall`` and ``Precheck``, but the two referenced
    each other's ids -- a two-way reference means they were one thing all along:
    a pitfall, and the assertion that stops it.
    """

    id: str
    mode: FailureMode
    symptom: str  # what it looks like
    actual_cause: str  # what it actually is
    guard: str  # the assertion that blocks it, one line
    guard_runs_at: NoiseFloor  # lowest floor that can run it -- cheap checks first


__all__ = ["FailureMode", "KnownPitfall"]
