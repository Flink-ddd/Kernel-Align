# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The execution pipeline: free functions only, no state.

One file per step, in the order they run::

    registry    plugin registration and factor discovery
    planner     filter, statically reject, expand variants, order by cost
    runner      execution loop, reuse decisions, variant repeats
    diagnosis   four gates plus the matrix
    report      filter false positives, trace root causes, emit the report

``comparison`` holds the declaration-driven contract comparison the runner uses.
"""

from rl_engine.mismatch.pipeline.comparison import compare_contracts, resolve_field_path
from rl_engine.mismatch.pipeline.diagnosis import CONVERGENCE_RATIO, diagnose
from rl_engine.mismatch.pipeline.planner import (
    ContradictoryFactor,
    UnmetPrerequisite,
    build_variants,
    missing_prerequisites,
    order_cases_by_rebind_cost,
    reject_contradictory_factors,
    suggested_floor_is_lowest,
)
from rl_engine.mismatch.pipeline.registry import (
    OPERATOR_CHECKS,
    FactorDiscoveryError,
    OperatorChecks,
    PluginRegistry,
    RegistrationError,
    discover_factors,
)
from rl_engine.mismatch.pipeline.report import (
    build_report,
    filter_known_equivalences,
    render_summary,
    trace_root_causes,
)
from rl_engine.mismatch.pipeline.runner import (
    ReadOnlyViolation,
    RunContext,
    ScoringBackend,
    assert_comparison_is_read_only,
    assert_order_is_topology_independent,
    compute_metrics,
    expand_repeats,
    run_variant,
)

__all__ = [
    "compare_contracts",
    "resolve_field_path",
    "CONVERGENCE_RATIO",
    "diagnose",
    "ContradictoryFactor",
    "UnmetPrerequisite",
    "build_variants",
    "missing_prerequisites",
    "order_cases_by_rebind_cost",
    "reject_contradictory_factors",
    "suggested_floor_is_lowest",
    "OPERATOR_CHECKS",
    "FactorDiscoveryError",
    "OperatorChecks",
    "PluginRegistry",
    "RegistrationError",
    "discover_factors",
    "build_report",
    "filter_known_equivalences",
    "render_summary",
    "trace_root_causes",
    "ReadOnlyViolation",
    "RunContext",
    "ScoringBackend",
    "assert_comparison_is_read_only",
    "assert_order_is_topology_independent",
    "compute_metrics",
    "expand_repeats",
    "run_variant",
]
