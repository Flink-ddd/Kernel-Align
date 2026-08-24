# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework integration entry points owned by RL-Kernel."""

from rl_engine.integrations.ablation import (
    Implementation,
    IntegrationPlan,
    OperatorAblationCase,
    operator_ablation_case,
    operator_ablation_cases,
)
from rl_engine.integrations.megatron import MegatronIntegration
from rl_engine.integrations.vllm import VllmIntegration

__all__ = [
    "Implementation",
    "IntegrationPlan",
    "MegatronIntegration",
    "OperatorAblationCase",
    "VllmIntegration",
    "operator_ablation_case",
    "operator_ablation_cases",
]
