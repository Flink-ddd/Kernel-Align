# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prometheus /metrics HTTP exporter for RL-Kernel.

The server is never started as an import side effect. Application code (or an
example honoring ``RL_KERNEL_METRICS=1``) must call ``start_metrics_server``
explicitly; library executors only record into the facade.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

from rl_engine.observability.metrics import metrics
from rl_engine.utils.logger import logger

RL_KERNEL_METRICS = "RL_KERNEL_METRICS"
RL_KERNEL_METRICS_PORT = "RL_KERNEL_METRICS_PORT"
RL_KERNEL_METRICS_ADDR = "RL_KERNEL_METRICS_ADDR"

_DEFAULT_PORT = 8000
_DEFAULT_ADDR = "0.0.0.0"

_lock = threading.Lock()
_bound_port: Optional[int] = None


def start_metrics_server(port: Optional[int] = None, addr: Optional[str] = None) -> int:
    """Start the /metrics HTTP server for this process and return the bound port.

    Idempotent per process: subsequent calls return the port of the already
    running server. ``port=0`` binds an ephemeral port. Defaults come from
    ``RL_KERNEL_METRICS_PORT`` / ``RL_KERNEL_METRICS_ADDR``.

    Raises RuntimeError when prometheus_client is not installed; use
    ``maybe_start_metrics_server_from_env`` for a best-effort opt-in start.
    """
    global _bound_port

    with _lock:
        if _bound_port is not None:
            logger.warn_once(
                "start_metrics_server called more than once; reusing the running exporter."
            )
            return _bound_port

        registry = metrics.collector_registry
        if registry is None:
            raise RuntimeError(
                "prometheus_client is required for the metrics exporter. "
                "Install it with `pip install rl-kernel[observability]`."
            )
        from prometheus_client import start_http_server

        resolved_port = (
            port
            if port is not None
            else int(os.environ.get(RL_KERNEL_METRICS_PORT, str(_DEFAULT_PORT)))
        )
        resolved_addr = addr or os.environ.get(RL_KERNEL_METRICS_ADDR, _DEFAULT_ADDR)

        server, _thread = start_http_server(resolved_port, addr=resolved_addr, registry=registry)
        _bound_port = int(server.server_port)
        logger.info(f"RL-Kernel metrics exporter listening on {resolved_addr}:{_bound_port}")
        return _bound_port


def maybe_start_metrics_server_from_env() -> Optional[int]:
    """Start the exporter iff ``RL_KERNEL_METRICS=1``; never raises.

    Returns the bound port, or None when disabled or unavailable. Intended for
    examples and entrypoints, not library code.
    """
    if os.environ.get(RL_KERNEL_METRICS, "").strip() != "1":
        return None
    try:
        return start_metrics_server()
    except Exception as exc:
        logger.warning(f"RL_KERNEL_METRICS=1 but the metrics exporter could not start: {exc}")
        return None
