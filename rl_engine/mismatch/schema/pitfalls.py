# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Known pitfalls, encoded as data.

Prose pitfalls get read once and never again. As data, the framework can block
them before a run instead of relying on somebody remembering afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rl_engine.mismatch.schema.variants import NoiseFloor


class FailureMode(str, Enum):
    """How a pitfall fools you, which decides what tool works against it."""

    STRUCTURAL_FALSE_POSITIVE = "structural_false_positive"  # differs in form, equal in math
    SILENT_FALSE_NEGATIVE = "silent_false_negative"  # metrics look fine, conclusion is wrong
    MISSING_INSTRUMENTATION = "missing_instrumentation"  # never captured in the first place
    CONFIG_DEFAULT_TRAP = "config_default_trap"  # the default is not what you assumed
    CONVENTION_MISMATCH = "convention_mismatch"  # shift-by-one, log base, ...
    RESOURCE_LIMIT = "resource_limit"  # this arm simply cannot run


@dataclass(frozen=True)
class KnownPitfall:
    """A known pitfall together with the assertion that blocks it.

    ``symptom`` and ``actual_cause`` are separate because a pitfall is a pitfall
    precisely when its appearance points at the wrong cause.
    """

    id: str
    mode: FailureMode
    symptom: str
    actual_cause: str
    guard: str
    guard_runs_at: NoiseFloor  # lowest floor that can run it -- cheap checks first


__all__ = ["FailureMode", "KnownPitfall"]
