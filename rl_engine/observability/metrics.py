# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Prometheus metrics facade for RL-Kernel.

The facade exposes domain-level recording methods instead of raw prometheus
objects. ``prometheus_client`` is an optional dependency: when it cannot be
imported every method degrades to a silent no-op, so engine code may call the
module-level ``metrics`` singleton unconditionally.
"""

from __future__ import annotations

import bisect
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from rl_engine.utils.logger import logger

# Metric families exported by this facade. tests/test_observability.py uses this
# schema to guard examples/grafana-dashboard.json against drift.
SCHEMA_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "rl_kernel_build_info",
        "rl_kernel_stage_duration_seconds",
        "rl_kernel_requests_total",
        "rl_kernel_output_tokens_total",
        "rl_kernel_selected_backend_info",
        "rl_kernel_hardware_fallback_total",
        "rl_kernel_kv_cache_blocks",
        "rl_kernel_kv_cache_fragmentation_ratio",
        "rl_kernel_gpu_peak_memory_bytes",
        "rl_kernel_training_loss",
        "rl_kernel_weight_version",
    }
)

# Stage latencies span sub-millisecond kernel paths to multi-minute rollouts.
_STAGE_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
)


@dataclass
class _PendingStageUpdate:
    """Bounded aggregate for amortized Prometheus child updates."""

    count: int = 0
    output_tokens: int = 0
    duration_sum: float = 0.0
    durations: list[float] = field(default_factory=list)
    bucket_counts: list[int] = field(
        default_factory=lambda: [0] * (len(_STAGE_DURATION_BUCKETS) + 1)
    )
    allocated_bytes: Optional[int] = None
    reserved_bytes: Optional[int] = None


class MetricsRegistry:
    """Prometheus facade. Every method is a safe no-op when prometheus_client is absent.

    Each instance owns a dedicated ``CollectorRegistry`` (never the process-global
    default) so instances cannot collide and tests stay isolated.
    """

    def __init__(self, *, stage_batch_size: int = 1) -> None:
        if stage_batch_size < 1:
            raise ValueError("stage_batch_size must be at least 1")
        self._initialized = False
        self._enabled = False
        self._registry: Any = None
        self._stage_batch_size = stage_batch_size
        self._pending_stage_updates: dict[tuple[str, str], _PendingStageUpdate] = {}
        self._pending_stage_lock = threading.Lock()
        # Resolving a labelled child costs a dict lookup plus a lock on every
        # call, so cache them: instrumented sites run per step and label values
        # come from finite enums, which bounds this cache by construction.
        self._children: dict[tuple[str, tuple[str, ...]], Any] = {}
        self._stage_duration: Any = None
        self._requests: Any = None
        self._output_tokens: Any = None
        self._selected_backend: Any = None
        self._hardware_fallback: Any = None
        self._kv_cache_blocks: Any = None
        self._kv_cache_fragmentation: Any = None
        self._gpu_peak_memory: Any = None
        self._training_loss: Any = None
        self._weight_version: Any = None

    @property
    def enabled(self) -> bool:
        """Whether prometheus_client is importable and collectors are registered."""
        return self._ensure_initialized()

    @property
    def collector_registry(self) -> Optional[Any]:
        """The dedicated CollectorRegistry, or None when disabled."""
        return self._registry if self._ensure_initialized() else None

    # ------------------------------------------------------------------
    # Domain-level recording methods
    # ------------------------------------------------------------------

    @contextmanager
    def stage_timer(self, stage: str) -> Iterator[None]:
        """Time a stage; records duration and a request with ok/error status.

        Exceptions propagate unchanged after being recorded with status="error".
        """
        start = time.perf_counter()
        try:
            yield
        except BaseException:
            self.record_stage(stage, time.perf_counter() - start, status="error")
            raise
        self.record_stage(stage, time.perf_counter() - start, status="ok")

    def record_stage(
        self,
        stage: str,
        duration_seconds: float,
        *,
        status: str,
        output_tokens: int = 0,
        allocated_bytes: Optional[int] = None,
        reserved_bytes: Optional[int] = None,
    ) -> None:
        """Record one completed stage through a single facade call.

        Executors that already measure their own stage duration should use this
        method instead of wrapping the same work in ``stage_timer``. Besides
        keeping the exported duration aligned with their result contract, the
        combined update avoids repeated initialization checks and labelled-child
        cache lookups on latency-sensitive paths.
        """
        if not self._ensure_initialized():
            return
        if self._stage_batch_size == 1:
            self._record_stage_now(
                stage,
                duration_seconds,
                status=status,
                output_tokens=output_tokens,
                allocated_bytes=allocated_bytes,
                reserved_bytes=reserved_bytes,
            )
            return

        key = (stage, status)
        with self._pending_stage_lock:
            pending = self._pending_stage_updates.get(key)
            if pending is None:
                pending = _PendingStageUpdate()
                self._pending_stage_updates[key] = pending
            pending.count += 1
            pending.output_tokens += max(0, output_tokens)
            pending.duration_sum += duration_seconds
            pending.durations.append(duration_seconds)
            bucket_index = bisect.bisect_left(_STAGE_DURATION_BUCKETS, duration_seconds)
            pending.bucket_counts[bucket_index] += 1
            if allocated_bytes is not None:
                pending.allocated_bytes = allocated_bytes
            if reserved_bytes is not None:
                pending.reserved_bytes = reserved_bytes
            if pending.count < self._stage_batch_size:
                return
            del self._pending_stage_updates[key]
        self._flush_stage_update(stage, status, pending)

    def flush(self) -> None:
        """Flush pending batched updates before collection or shutdown."""
        if not self._ensure_initialized():
            return
        with self._pending_stage_lock:
            pending_updates = self._pending_stage_updates
            self._pending_stage_updates = {}
        for (stage, status), pending in pending_updates.items():
            self._flush_stage_update(stage, status, pending)

    def add_output_tokens(self, stage: str, count: int) -> None:
        if not self._ensure_initialized() or count <= 0:
            return
        self._child(self._output_tokens, "tokens", (stage,)).inc(count)

    def set_selected_backend(self, op_type: str, backend: str, *, is_fallback: bool) -> None:
        if not self._ensure_initialized():
            return
        labels = (op_type, backend, str(is_fallback).lower())
        self._child(self._selected_backend, "backend", labels).set(1)

    def record_hardware_fallback(
        self,
        op_type: str,
        *,
        requested_backend: str,
        selected_backend: str,
        reason: str,
    ) -> None:
        if not self._ensure_initialized():
            return
        labels = (op_type, requested_backend, selected_backend, reason)
        self._child(self._hardware_fallback, "fallback", labels).inc()

    def set_kv_cache_blocks(self, *, required: int, reserved: int) -> None:
        if not self._ensure_initialized():
            return
        self._child(self._kv_cache_blocks, "kv", ("required",)).set(required)
        self._child(self._kv_cache_blocks, "kv", ("reserved",)).set(reserved)
        fragmentation = 0.0 if reserved <= 0 else 1.0 - (required / reserved)
        self._kv_cache_fragmentation.set(fragmentation)

    def set_gpu_peak_memory(
        self,
        stage: str,
        *,
        allocated_bytes: int,
        reserved_bytes: int,
    ) -> None:
        if not self._ensure_initialized():
            return
        self._child(self._gpu_peak_memory, "mem", (stage, "allocated")).set(allocated_bytes)
        self._child(self._gpu_peak_memory, "mem", (stage, "reserved")).set(reserved_bytes)

    def set_training_loss(self, loss: float) -> None:
        if not self._ensure_initialized():
            return
        self._training_loss.set(loss)

    def set_weight_version(self, role: str, version: int) -> None:
        if not self._ensure_initialized():
            return
        self._child(self._weight_version, "weights", (role,)).set(version)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _record_stage_now(
        self,
        stage: str,
        duration_seconds: float,
        *,
        status: str,
        output_tokens: int,
        allocated_bytes: Optional[int],
        reserved_bytes: Optional[int],
    ) -> None:
        self._child(self._stage_duration, "duration", (stage,)).observe(duration_seconds)
        self._child(self._requests, "requests", (stage, status)).inc()
        if output_tokens > 0:
            self._child(self._output_tokens, "tokens", (stage,)).inc(output_tokens)
        if allocated_bytes is not None:
            self._child(self._gpu_peak_memory, "mem", (stage, "allocated")).set(
                allocated_bytes
            )
        if reserved_bytes is not None:
            self._child(self._gpu_peak_memory, "mem", (stage, "reserved")).set(reserved_bytes)

    def _flush_stage_update(
        self,
        stage: str,
        status: str,
        pending: _PendingStageUpdate,
    ) -> None:
        duration = self._child(self._stage_duration, "duration", (stage,))
        # prometheus_client has no public batch-observe API. Its Histogram child
        # stores the non-cumulative bucket values and sum as thread-safe Value
        # objects; incrementing those once per batch preserves exactly the same
        # samples as calling observe() for every stage, while moving locks and
        # label lookups off the hot path.
        histogram_sum = getattr(duration, "_sum", None)
        histogram_buckets = getattr(duration, "_buckets", None)
        if (
            histogram_sum is not None
            and histogram_buckets is not None
            and len(histogram_buckets) == len(pending.bucket_counts)
        ):
            histogram_sum.inc(pending.duration_sum)
            for bucket, count in zip(
                histogram_buckets,
                pending.bucket_counts,
                strict=True,
            ):
                if count:
                    bucket.inc(count)
        else:  # preserve correctness if prometheus_client changes its internals
            for duration_seconds in pending.durations:
                duration.observe(duration_seconds)
        self._child(self._requests, "requests", (stage, status)).inc(pending.count)
        if pending.output_tokens > 0:
            self._child(self._output_tokens, "tokens", (stage,)).inc(pending.output_tokens)
        if pending.allocated_bytes is not None:
            self._child(self._gpu_peak_memory, "mem", (stage, "allocated")).set(
                pending.allocated_bytes
            )
        if pending.reserved_bytes is not None:
            self._child(self._gpu_peak_memory, "mem", (stage, "reserved")).set(
                pending.reserved_bytes
            )

    def _child(self, metric: Any, key: str, labels: tuple[str, ...]) -> Any:
        """Return a labelled child, caching it across calls."""
        cache_key = (key, labels)
        child = self._children.get(cache_key)
        if child is None:
            child = metric.labels(*labels)
            self._children[cache_key] = child
        return child

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._enabled
        self._initialized = True
        try:
            import prometheus_client
        except ImportError:
            logger.info_once(
                "prometheus_client is not installed; RL-Kernel metrics are disabled. "
                "Install with `pip install rl-kernel[observability]` to enable them."
            )
            self._enabled = False
            return False
        try:
            self._build_collectors(prometheus_client)
            self._enabled = True
        except Exception as exc:  # exporter failures must never break engine code
            logger.warning(f"Failed to initialize RL-Kernel metrics collectors: {exc}")
            self._enabled = False
        return self._enabled

    def _build_collectors(self, prometheus_client: Any) -> None:
        owner = self

        class _FlushingCollectorRegistry(prometheus_client.CollectorRegistry):
            def collect(registry_self) -> Iterator[Any]:
                owner.flush()
                yield from super().collect()

        registry = _FlushingCollectorRegistry()
        self._registry = registry

        for collector_name in ("ProcessCollector", "PlatformCollector"):
            collector_cls = getattr(prometheus_client, collector_name, None)
            if collector_cls is None:
                continue
            try:
                collector_cls(registry=registry)
            except Exception:
                # Platform/process collectors are best-effort (e.g. non-Linux).
                pass

        self._stage_duration = prometheus_client.Histogram(
            "rl_kernel_stage_duration_seconds",
            "Wall-clock duration of an instrumented RL-Kernel stage.",
            labelnames=("stage",),
            buckets=_STAGE_DURATION_BUCKETS,
            registry=registry,
        )
        self._requests = prometheus_client.Counter(
            "rl_kernel_requests",
            "Completed stage executions by status.",
            labelnames=("stage", "status"),
            registry=registry,
        )
        self._output_tokens = prometheus_client.Counter(
            "rl_kernel_output_tokens",
            "Tokens processed per stage (generated for rollout, active for train/score).",
            labelnames=("stage",),
            registry=registry,
        )
        self._selected_backend = prometheus_client.Gauge(
            "rl_kernel_selected_backend_info",
            "Kernel backend selected at dispatch time (value is always 1).",
            labelnames=("op_type", "backend", "is_fallback"),
            registry=registry,
        )
        self._hardware_fallback = prometheus_client.Counter(
            "rl_kernel_hardware_fallback",
            "Runtime fallback events from a requested backend to a selected backend.",
            labelnames=("op_type", "requested_backend", "selected_backend", "reason"),
            registry=registry,
        )
        self._kv_cache_blocks = prometheus_client.Gauge(
            "rl_kernel_kv_cache_blocks",
            "Paged KV-cache blocks by kind.",
            labelnames=("kind",),
            registry=registry,
        )
        self._kv_cache_fragmentation = prometheus_client.Gauge(
            "rl_kernel_kv_cache_fragmentation_ratio",
            "Paged KV-cache fragmentation: 1 - required/reserved (0 when reserved is 0).",
            registry=registry,
        )
        self._gpu_peak_memory = prometheus_client.Gauge(
            "rl_kernel_gpu_peak_memory_bytes",
            "Peak GPU memory observed during a stage.",
            labelnames=("stage", "kind"),
            registry=registry,
        )
        self._training_loss = prometheus_client.Gauge(
            "rl_kernel_training_loss",
            "Most recent training loss.",
            registry=registry,
        )
        self._weight_version = prometheus_client.Gauge(
            "rl_kernel_weight_version",
            "Latest published/consumed weight version.",
            labelnames=("role",),
            registry=registry,
        )
        self._set_build_info(prometheus_client, registry)

    def _set_build_info(self, prometheus_client: Any, registry: Any) -> None:
        build_info = prometheus_client.Gauge(
            "rl_kernel_build_info",
            "RL-Kernel build/platform information (value is always 1).",
            labelnames=("version", "device_type", "backend_version", "ext_available"),
            registry=registry,
        )
        build_info.labels(
            version=_package_version(),
            device_type=_device_label("device_type"),
            backend_version=_device_label("backend_version"),
            ext_available=str(_extension_available()).lower(),
        ).set(1)


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("RL-Kernel")
    except Exception:
        return "unknown"


def _device_label(attribute: str) -> str:
    try:
        from rl_engine.platforms.device import device_ctx

        return str(getattr(device_ctx, attribute))
    except Exception:
        return "unknown"


def _extension_available() -> bool:
    try:
        from rl_engine.kernels.ops.base import _EXT_AVAILABLE

        return bool(_EXT_AVAILABLE)
    except Exception:
        return False


metrics = MetricsRegistry(stage_batch_size=16)  # module-level singleton, mirrors utils/logger.py
