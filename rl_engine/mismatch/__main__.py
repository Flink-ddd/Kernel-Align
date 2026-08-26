# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Command line entry point, and the only place that imports operator plugins.

Importing them here is what triggers self-registration, and it keeps the
dependency arrow pointing one way: ``pipeline/`` never reaches into
``operator_checks/``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Sequence

from rl_engine.mismatch.pipeline import (
    OPERATOR_CHECKS,
    build_variants,
    missing_prerequisites,
    order_cases_by_rebind_cost,
    reject_contradictory_factors,
)
from rl_engine.mismatch.schema import NoiseFloor

# An operator that is not listed here does not exist as far as the framework
# is concerned.
_OPERATOR_PACKAGES: tuple[str, ...] = (
    "rl_engine.mismatch.operator_checks.gemm",
    "rl_engine.mismatch.operator_checks.attention",
    "rl_engine.mismatch.operator_checks.logprob",
)


def load_operator_plugins(packages: Sequence[str] = _OPERATOR_PACKAGES) -> tuple[str, ...]:
    """Import each plugin package so its decorator runs."""

    for package in packages:
        importlib.import_module(package)
    return OPERATOR_CHECKS.operators()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_engine.mismatch",
        description="Diagnose training-inference mismatch, one factor at a time.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="list registered operators and their factors")
    listing.add_argument("--operator", default=None)

    plan = sub.add_parser("plan", help="expand factors into variants without running anything")
    plan.add_argument("--operator", default=None)
    plan.add_argument(
        "--noise-floor",
        default=NoiseFloor.SINGLE_LAYER_ANCHOR.value,
        choices=[floor.value for floor in NoiseFloor],
    )
    plan.add_argument("--gpu-count", type=int, default=0)
    plan.add_argument("--json", action="store_true")

    return parser


def command_list(operator: str | None) -> int:
    operators = load_operator_plugins()
    if not operators:
        print(
            "no operator plugins registered.\n"
            "The framework ships without operators: add one under "
            "rl_engine/mismatch/operator_checks/<operator>/ and list it in "
            "__main__._OPERATOR_PACKAGES.\n"
            "See rl_engine/mismatch/operator_checks/__init__.py for the layout."
        )
        return 0

    for name in operators:
        if operator is not None and name != operator:
            continue
        factors = OPERATOR_CHECKS.factors_for(name)
        print(f"{name}: {len(factors)} factors")
        for factor in factors:
            print(f"  {factor.id:<40} {factor.category.value}")
    return 0


def command_plan(operator: str | None, noise_floor: str, gpu_count: int, as_json: bool) -> int:
    load_operator_plugins()
    factors = OPERATOR_CHECKS.factors_for(operator)
    if not factors:
        print("nothing to plan: no operator plugins are registered.")
        return 0

    reject_contradictory_factors(factors)

    runnable = []
    skipped = []
    for factor in factors:
        unmet = missing_prerequisites(factor, gpu_count=gpu_count)
        if unmet:
            skipped.append((factor, unmet))
        else:
            runnable.append(factor)

    cases = [(factor, variant) for factor in runnable for variant in build_variants(factor)]
    ordered = order_cases_by_rebind_cost(cases)

    if as_json:
        payload = {
            "noise_floor": noise_floor,
            "runnable_factors": [factor.id for factor in runnable],
            "skipped": {factor.id: [item.reason for item in unmet] for factor, unmet in skipped},
            "cases": [
                {
                    "factor": factor.id,
                    "variant": variant.name,
                    "rebind_cost": factor.switch.rebind_cost.value,
                    "switch_values": dict(variant.switch_values),
                }
                for factor, variant in ordered
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"noise floor: {noise_floor}")
    print(f"runnable factors: {len(runnable)}   cases: {len(ordered)}")
    if skipped:
        print("\nskipped (prerequisites not met):")
        for factor, unmet in skipped:
            for item in unmet:
                print(f"  {factor.id}: {item.reason}")
    print("\ncases in execution order (cheapest rebuild first):")
    for factor, variant in ordered:
        print(f"  [{factor.switch.rebind_cost.value:<22}] {factor.id} :: {variant.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return command_list(args.operator)
    if args.command == "plan":
        return command_plan(args.operator, args.noise_floor, args.gpu_count, args.json)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
