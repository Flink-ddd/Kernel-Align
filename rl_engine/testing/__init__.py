# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Testing helpers for RL-shaped kernel validation."""

from .reference_ops import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    masked_mean,
    masked_sum,
    selected_logprobs_reference,
    summarize_kernel_drift,
)
from .rl_batch import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch
from .ws1_workload import (
    LogicalBatch,
    PhysicalLayout,
    WorkloadError,
    WS1Manifest,
    apply_chunking,
    apply_packing,
    apply_padding,
    build_logical_batch,
    fixture_hash,
    load_manifest,
    reference_payload,
    restore_logical_order,
    restore_logical_order_from_padded,
)

__all__ = [
    "LogicalBatch",
    "PhysicalLayout",
    "SyntheticRLKernelBatch",
    "WS1Manifest",
    "WorkloadError",
    "active_token_count",
    "apply_chunking",
    "apply_padding",
    "apply_packing",
    "build_logical_batch",
    "compute_policy_ratio",
    "compute_reference_kl",
    "fixture_hash",
    "load_manifest",
    "make_synthetic_rl_kernel_batch",
    "masked_mean",
    "masked_sum",
    "reference_payload",
    "restore_logical_order",
    "restore_logical_order_from_padded",
    "selected_logprobs_reference",
    "summarize_kernel_drift",
]
