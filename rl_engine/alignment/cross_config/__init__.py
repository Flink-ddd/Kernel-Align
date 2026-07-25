# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plan and run cross-configuration alignment experiments.

The package root is intentionally small and lazily loads execution code. Extension
authors import adapter, artifact, operator, or schema details from their owning
submodule.
"""

from importlib import import_module
from typing import Any

from rl_engine.alignment.cross_config.comparison import compare_score_artifacts
from rl_engine.alignment.cross_config.config import ExperimentConfig, load_config
from rl_engine.alignment.cross_config.execution_plan import ExecutionPlan, build_execution_plan
from rl_engine.alignment.cross_config.planner import ExperimentPlan, Planner
from rl_engine.alignment.cross_config.standard import (
    ALIGNMENT_PROFILE_VERSION,
    ALIGNMENT_STANDARD_ID,
    DISTRIBUTED_ALIGNMENT_PROFILES,
    AlignmentProfile,
    AlignmentStandard,
    RLK_ALIGNMENT_PROFILE_ORDER,
    alignment_profile_fingerprint,
    get_alignment_standard,
    iter_alignment_profiles,
)


def __getattr__(name: str) -> Any:
    if name in {"PairedRunResult", "PairedRunner"}:
        return getattr(import_module("rl_engine.alignment.cross_config.runner"), name)
    if name in {"RuntimeMaterializer", "RuntimeTools"}:
        return getattr(import_module("rl_engine.alignment.cross_config.runtime"), name)
    raise AttributeError(name)


__all__ = [
    "ExperimentConfig",
    "ExperimentPlan",
    "ExecutionPlan",
    "ALIGNMENT_PROFILE_VERSION",
    "ALIGNMENT_STANDARD_ID",
    "DISTRIBUTED_ALIGNMENT_PROFILES",
    "AlignmentProfile",
    "AlignmentStandard",
    "PairedRunResult",
    "PairedRunner",
    "Planner",
    "RLK_ALIGNMENT_PROFILE_ORDER",
    "RuntimeMaterializer",
    "RuntimeTools",
    "alignment_profile_fingerprint",
    "build_execution_plan",
    "compare_score_artifacts",
    "get_alignment_standard",
    "iter_alignment_profiles",
    "load_config",
]
