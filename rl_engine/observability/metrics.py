# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prometheus metrics for RL-Kernel: op throughput, backend-fallback rate, and
KV-cache fragmentation, exposed over an opt-in `/metrics` HTTP endpoint.

`prometheus_client` is an optional dependency (install via `pip install -e .[observability]`).
Every public function in this module degrades to a safe no-op when it is unavailable, mirroring
the `_C`/`_EXT_AVAILABLE` optional-extension pattern in `rl_engine/kernels/ops/base.py`.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from rl_engine.utils.logger import logger

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    _PROMETHEUS_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"prometheus_client unavailable: {e}. Observability metrics will be no-ops. "
        "Install with `pip install -e .[observability]` to enable them."
    )
    _PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = start_http_server = None  # type: ignore[assignment]

_OP_METRICS_ENV = "RL_KERNEL_ENABLE_OP_METRICS"
_METRICS_SERVER_ENV = "RL_KERNEL_ENABLE_METRICS_SERVER"
_METRICS_PORT_ENV = "RL_KERNEL_METRICS_PORT"
_DEFAULT_METRICS_PORT = 9400

_TRUE_VALUES = {"1", "true", "yes", "on"}


_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    logger.warn_once(
        f"{name}={value!r} is not a recognized boolean flag "
        f"(expected one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}); using default={default}."
    )
    return default


def metrics_enabled() -> bool:
    """Whether the always-on recorders (fallback, KV-cache fragmentation) are active.

    These recorders never change any caller's return type or value, so they are gated only on
    dependency availability -- free once `prometheus_client` is installed.
    """
    return _PROMETHEUS_AVAILABLE


def op_metrics_enabled() -> bool:
    """Whether `KernelRegistry.get_op(...)` should wrap its return value in `InstrumentedOp`.

    Defaults OFF and requires explicit opt-in via `RL_KERNEL_ENABLE_OP_METRICS=1`, separately
    from `metrics_enabled()`, because wrapping changes `get_op(...)`'s return type: several
    existing tests assert `isinstance(kernel_registry.get_op(op_type), ConcreteOpClass)` (e.g.
    tests/test_embedding.py, tests/test_swiglu.py, tests/test_lm_head.py, tests/test_attention.py,
    tests/test_kv_cache_attention.py), and wrapping by default would break all of them.
    """
    return _PROMETHEUS_AVAILABLE and _env_flag(_OP_METRICS_ENV, default=False)


def metrics_server_enabled() -> bool:
    """Whether a worker should auto-start the `/metrics` HTTP endpoint on init."""
    return _env_flag(_METRICS_SERVER_ENV, default=False)


if _PROMETHEUS_AVAILABLE:
    OP_CALLS_TOTAL = Counter(
        "rlkernel_op_calls_total",
        "Total kernel operator invocations.",
        ["op_type", "backend", "method"],
    )
    OP_LATENCY_SECONDS = Histogram(
        "rlkernel_op_latency_seconds",
        "Kernel operator invocation latency, seconds.",
        ["op_type", "backend", "method"],
    )
    OP_FALLBACKS_TOTAL = Counter(
        "rlkernel_op_fallbacks_total",
        "Times a preferred backend failed to load/instantiate and dispatch fell back.",
        ["op_type", "failed_backend"],
    )
    KV_CACHE_FRAGMENTATION_RATIO = Gauge(
        "rlkernel_kv_cache_fragmentation_ratio",
        "1 - (required_blocks / reserved_blocks) for the most recent paged-KV reservation.",
        ["baseline_kind"],
    )
else:
    OP_CALLS_TOTAL = None
    OP_LATENCY_SECONDS = None
    OP_FALLBACKS_TOTAL = None
    KV_CACHE_FRAGMENTATION_RATIO = None


def record_fallback(op_type: str, failed_backend: str) -> None:
    """Record that `failed_backend` could not serve `op_type` and dispatch moved on."""
    if not metrics_enabled():
        return
    OP_FALLBACKS_TOTAL.labels(op_type=op_type, failed_backend=failed_backend).inc()


def record_kv_cache_fragmentation(
    required_blocks: int, reserved_blocks: int, *, baseline_kind: str
) -> None:
    """Record the reserved-but-unused fraction of a paged-KV-cache block reservation."""
    if not metrics_enabled() or reserved_blocks <= 0:
        return
    fragmentation = 1.0 - (required_blocks / reserved_blocks)
    KV_CACHE_FRAGMENTATION_RATIO.labels(baseline_kind=baseline_kind).set(fragmentation)


class InstrumentedOp:
    """Transparent latency/throughput proxy around a resolved kernel op instance.

    Wraps whatever `KernelRegistry.get_op(...)` resolves. `__call__` and every callable
    attribute access (`.forward`, `.forward_fp32`, ...) are timed under the same
    `(op_type, backend)` label pair plus a `method` label, so the underlying op classes need no
    changes. Non-callable attributes (e.g. `.op_class`) pass through untouched.

    Does not fake `isinstance` against the wrapped type -- callers needing type identity should
    read `.op_class` or avoid opting into `RL_KERNEL_ENABLE_OP_METRICS`.
    """

    def __init__(self, wrapped: Any, *, op_type: str, backend: str):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_op_type", op_type)
        object.__setattr__(self, "_backend", backend)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._timed(self._wrapped, "__call__", args, kwargs)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._wrapped, name)
        if not callable(attr):
            return attr

        def _call(*args: Any, __attr: Any = attr, __name: str = name, **kwargs: Any) -> Any:
            return self._timed(__attr, __name, args, kwargs)

        return _call

    def _timed(self, fn: Any, method: str, args: tuple, kwargs: dict) -> Any:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            OP_CALLS_TOTAL.labels(op_type=self._op_type, backend=self._backend, method=method).inc()
            OP_LATENCY_SECONDS.labels(
                op_type=self._op_type, backend=self._backend, method=method
            ).observe(elapsed)


_server_started = False


def start_metrics_server(port: Optional[int] = None) -> Optional[int]:
    """Idempotent per-process `/metrics` HTTP server start.

    Wraps `prometheus_client.start_http_server` (stdlib `http.server` based, so this adds no
    Flask/FastAPI dependency). Returns the bound port, or None if metrics are unavailable or the
    server is already running in this process.
    """
    global _server_started
    if not metrics_enabled() or _server_started:
        return None
    if port is None:
        try:
            base_port = int(os.environ.get(_METRICS_PORT_ENV, str(_DEFAULT_METRICS_PORT)))
        except (TypeError, ValueError):
            logger.warn_once(
                "%s=%r is not a valid integer port; using default=%d.",
                _METRICS_PORT_ENV,
                os.environ.get(_METRICS_PORT_ENV),
                _DEFAULT_METRICS_PORT,
            )
            base_port = _DEFAULT_METRICS_PORT
        try:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        except (TypeError, ValueError):
            logger.warn_once(
                "RANK/LOCAL_RANK env var is not a valid integer; using rank=0.",
            )
            rank = 0
        port = base_port + rank
    try:
        start_http_server(port)
    except OSError as e:
        # Observability must never take down real training/rollout work -- a bound port,
        # a sandboxed network namespace, etc. should degrade to "no /metrics", not a crash.
        logger.warning(f"Failed to start Prometheus /metrics server on :{port}: {e}")
        return None
    _server_started = True
    logger.info(f"Prometheus /metrics server listening on :{port}")
    return port
