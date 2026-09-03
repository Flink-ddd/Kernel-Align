# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework integration entry points owned by RL-Kernel."""

from rl_engine.integrations.ablation import (
    Implementation,
    IntegrationPlan,
    OperatorAblationCase,
    configure_integration_environment,
    integration_plan_from_environment,
    operator_ablation_case,
    operator_ablation_cases,
)
from rl_engine.integrations.megatron import MegatronIntegration
from rl_engine.integrations.megatron_runtime import install_megatron_integration
from rl_engine.integrations.rocm_ablation import (
    ROCM_ATTENTION_CASE_IDS,
    RocmAblationCaseResult,
    RocmAttentionAblationCase,
    rocm_attention_ablation_matrix,
    run_rocm_attention_ablation,
)
from rl_engine.integrations.vllm import VllmIntegration
from rl_engine.integrations.vllm_runtime import configure_vllm_environment

__all__ = [
    "Implementation",
    "IntegrationPlan",
    "MegatronIntegration",
    "OperatorAblationCase",
    "ROCM_ATTENTION_CASE_IDS",
    "RocmAblationCaseResult",
    "RocmAttentionAblationCase",
    "VllmIntegration",
    "configure_integration_environment",
    "configure_vllm_environment",
    "install_megatron_integration",
    "integration_plan_from_environment",
    "operator_ablation_case",
    "operator_ablation_cases",
    "rocm_attention_ablation_matrix",
    "run_rocm_attention_ablation",
]
