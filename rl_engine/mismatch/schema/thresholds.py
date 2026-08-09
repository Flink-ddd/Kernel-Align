# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The threshold table, as code constants.

Thresholds are not configuration. A tunable threshold means somebody can tune it
until the test passes, so this table lives in code, goes into the execution
fingerprint, and changing it invalidates historical pass/fail results.
"""

from __future__ import annotations

from rl_engine.mismatch.schema.variants import ExpectedRange, NoiseFloor

ANY_MODEL_FAMILY = "*"


EXPECTED_RANGES: tuple[ExpectedRange, ...] = (
    # -- production floor, empirical --
    ExpectedRange("dense", NoiseFloor.PRODUCTION, None, (0.002, 0.008), 0.01),
    ExpectedRange("moe", NoiseFloor.PRODUCTION, False, (0.007, 0.008), 0.02),
    ExpectedRange(
        "moe",
        NoiseFloor.PRODUCTION,
        True,
        (0.0, 0.008),
        0.008,
        note="routing replay on but no drop means replay never took effect",
    ),
    ExpectedRange(
        "large_moe",
        NoiseFloor.PRODUCTION,
        False,
        (0.01, 0.03),
        0.05,
        note="GLM5 / DSv3.2, DSA+MoE, 2k-8k. This band is normal, do not file it as a bug",
    ),
    ExpectedRange(
        "large_moe",
        NoiseFloor.PRODUCTION,
        True,
        (0.0, 0.008),
        0.008,
        note="no fall back to e-3 means the routing capture path is broken",
    ),
    # -- low-noise floors: the expectation is bitwise, not "small" --
    ExpectedRange(
        ANY_MODEL_FAMILY,
        NoiseFloor.SINGLE_LAYER_ANCHOR,
        None,
        (0.0, 0.0),
        1e-6,
        note="failing here is an operator bug, not training-inference mismatch",
    ),
    ExpectedRange(ANY_MODEL_FAMILY, NoiseFloor.FULL_MODEL_SINGLE_GPU, None, (0.0, 1e-6), 1e-5),
    ExpectedRange(ANY_MODEL_FAMILY, NoiseFloor.SHARDED_SINGLE_NODE, None, (0.0, 1e-4), 1e-3),
)


class ThresholdLookupError(LookupError):
    """No band declared for this combination."""


def expected_range(
    model_family: str,
    noise_floor: NoiseFloor,
    routing_replay: bool | None = None,
) -> ExpectedRange:
    """Look up the normal band for one combination.

    Reading a low floor with the production table hides real bugs: at
    SINGLE_LAYER_ANCHOR the expectation is bitwise, and judging it against
    0.002-0.008 would call a definite operator error "normal". So the lookup is
    always keyed by noise floor, and an exact model family beats the wildcard.
    """

    exact = [
        candidate
        for candidate in EXPECTED_RANGES
        if candidate.model_family == model_family
        and candidate.noise_floor is noise_floor
        and candidate.routing_replay == routing_replay
    ]
    if exact:
        return exact[0]

    wildcard = [
        candidate
        for candidate in EXPECTED_RANGES
        if candidate.model_family == ANY_MODEL_FAMILY and candidate.noise_floor is noise_floor
    ]
    if wildcard:
        return wildcard[0]

    raise ThresholdLookupError(
        f"no expected range declared for model_family={model_family!r} "
        f"noise_floor={noise_floor.value!r} routing_replay={routing_replay!r}"
    )


def tolerance_floor(model_family: str, noise_floor: NoiseFloor) -> float:
    """The absolute floor below which a difference is not treated as a signal.

    Used by the diagnosis matrix as ``tol_floor`` when deciding convergence.
    """

    return expected_range(model_family, noise_floor).suspect_above


__all__ = [
    "ANY_MODEL_FAMILY",
    "EXPECTED_RANGES",
    "ThresholdLookupError",
    "expected_range",
    "tolerance_floor",
]
