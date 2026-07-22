# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RL-Kernel observability: Prometheus metrics facade + NVTX stage ranges.

Importing this package has no side effects: nothing is registered globally and
no HTTP server starts until ``start_metrics_server`` is called explicitly.
"""

from rl_engine.observability.metrics import SCHEMA_METRIC_NAMES, MetricsRegistry, metrics
from rl_engine.observability.nvtx import nvtx_range
from rl_engine.observability.server import maybe_start_metrics_server_from_env, start_metrics_server

__all__ = [
    "SCHEMA_METRIC_NAMES",
    "MetricsRegistry",
    "metrics",
    "nvtx_range",
    "maybe_start_metrics_server_from_env",
    "start_metrics_server",
]
