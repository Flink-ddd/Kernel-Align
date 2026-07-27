# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-only tests for rl_engine.observability.metrics.

NVTX correctness (whether csrc/ops.cpp's ranges render as labeled, distinct blocks in an
`nsys` timeline) is NOT covered here and is not unit-testable in CI without a GPU + nsys. That
is a manual verification step documented in docs/getting_started/nsys-profiling.md, not a
Python-testable concern. It has been manually verified on real H100 hardware, including that
the RAII NVTX range still closes correctly when the wrapped op raises (`nsys stats
--report nvtx_sum` shows matched, non-orphaned ranges for both successful and raising calls).
"""

from __future__ import annotations

import socket
import urllib.request

import pytest
import torch

import rl_engine.observability.metrics as metrics_module
from rl_engine.executors.paged_kv_baseline import (
    PagedKVScoringConfig,
    collect_paged_kv_metrics,
    reserve_paged_kv_cache,
)
from rl_engine.executors.stateless_executor import StatelessForwardInputs, TensorTreeSummary
from rl_engine.kernels.registry import KernelRegistry, OpBackend
from rl_engine.observability.metrics import (
    InstrumentedOp,
    metrics_enabled,
    op_metrics_enabled,
    record_fallback,
    record_kv_cache_fragmentation,
    start_metrics_server,
)


def _sample_value(metric, name: str, **labels) -> float | None:
    """Read a Prometheus sample by exact name + labels via the public collect() API."""
    for family in metric.collect():
        for sample in family.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return None


def test_metrics_noop_without_prometheus(monkeypatch):
    monkeypatch.setattr(metrics_module, "_PROMETHEUS_AVAILABLE", False)

    assert metrics_enabled() is False
    assert op_metrics_enabled() is False

    # Must not raise even though the underlying metric objects are unusable stand-ins.
    record_fallback("some_op", "SOME_BACKEND")
    record_kv_cache_fragmentation(1, 2, baseline_kind="unit_test")
    assert start_metrics_server() is None


def test_op_metrics_enabled_requires_explicit_opt_in(monkeypatch):
    pytest.importorskip("prometheus_client")
    monkeypatch.delenv("RL_KERNEL_ENABLE_OP_METRICS", raising=False)
    assert op_metrics_enabled() is False

    monkeypatch.setenv("RL_KERNEL_ENABLE_OP_METRICS", "1")
    assert op_metrics_enabled() is True

    monkeypatch.setenv("RL_KERNEL_ENABLE_OP_METRICS", "0")
    assert op_metrics_enabled() is False


class _DummyOp:
    op_class = "dummy"

    def __call__(self, x: int) -> int:
        return x + 1

    def forward(self, x: int) -> int:
        return x * 2


def test_instrumented_op_records_calls_and_latency():
    pytest.importorskip("prometheus_client")
    op_type, backend = "unit_test_dummy_op", "UNIT_TEST_DUMMY_BACKEND"
    wrapped = InstrumentedOp(_DummyOp(), op_type=op_type, backend=backend)

    # Non-callable attributes pass through untouched.
    assert wrapped.op_class == "dummy"

    before_call = (
        _sample_value(
            metrics_module.OP_CALLS_TOTAL,
            "rlkernel_op_calls_total",
            op_type=op_type,
            backend=backend,
            method="__call__",
        )
        or 0.0
    )
    before_forward = (
        _sample_value(
            metrics_module.OP_CALLS_TOTAL,
            "rlkernel_op_calls_total",
            op_type=op_type,
            backend=backend,
            method="forward",
        )
        or 0.0
    )

    assert wrapped(3) == 4
    assert wrapped.forward(3) == 6

    after_call = _sample_value(
        metrics_module.OP_CALLS_TOTAL,
        "rlkernel_op_calls_total",
        op_type=op_type,
        backend=backend,
        method="__call__",
    )
    after_forward = _sample_value(
        metrics_module.OP_CALLS_TOTAL,
        "rlkernel_op_calls_total",
        op_type=op_type,
        backend=backend,
        method="forward",
    )
    latency_count = _sample_value(
        metrics_module.OP_LATENCY_SECONDS,
        "rlkernel_op_latency_seconds_count",
        op_type=op_type,
        backend=backend,
        method="__call__",
    )

    assert after_call == before_call + 1
    assert after_forward == before_forward + 1
    assert latency_count is not None and latency_count >= 1


def test_registry_fallback_increments_counter(monkeypatch):
    pytest.importorskip("prometheus_client")

    registry = KernelRegistry()
    op_type = "unit_test_fallback_op_type"
    failing_backend = OpBackend.PYTORCH_NATIVE
    working_backend = OpBackend.PYTORCH_NATIVE_SILU
    for platform in ("cpu", "cuda", "rocm"):
        registry._priority_map[platform][op_type] = [failing_backend, working_backend]

    original_load_backend = KernelRegistry._load_backend

    def fake_load_backend(self, backend):
        if backend is failing_backend:
            return None
        return original_load_backend(self, backend)

    monkeypatch.setattr(KernelRegistry, "_load_backend", fake_load_backend)

    before = (
        _sample_value(
            metrics_module.OP_FALLBACKS_TOTAL,
            "rlkernel_op_fallbacks_total",
            op_type=op_type,
            failed_backend=failing_backend.name,
        )
        or 0.0
    )

    op = registry.get_op(op_type)

    after = _sample_value(
        metrics_module.OP_FALLBACKS_TOTAL,
        "rlkernel_op_fallbacks_total",
        op_type=op_type,
        failed_backend=failing_backend.name,
    )

    assert op is not None
    assert after == before + 1


def test_registry_get_op_returns_raw_instance_by_default(monkeypatch):
    monkeypatch.delenv("RL_KERNEL_ENABLE_OP_METRICS", raising=False)

    registry = KernelRegistry()
    registry._priority_map["cpu"]["silu"] = [OpBackend.PYTORCH_NATIVE_SILU]
    registry._priority_map["cuda"]["silu"] = [OpBackend.PYTORCH_NATIVE_SILU]
    registry._priority_map["rocm"]["silu"] = [OpBackend.PYTORCH_NATIVE_SILU]

    op = registry.get_op("silu")

    assert not isinstance(op, InstrumentedOp)


def test_registry_get_op_wraps_when_op_metrics_enabled(monkeypatch):
    pytest.importorskip("prometheus_client")
    monkeypatch.setenv("RL_KERNEL_ENABLE_OP_METRICS", "1")

    registry = KernelRegistry()
    registry._priority_map["cpu"]["silu"] = [OpBackend.PYTORCH_NATIVE_SILU]
    registry._priority_map["cuda"]["silu"] = [OpBackend.PYTORCH_NATIVE_SILU]
    registry._priority_map["rocm"]["silu"] = [OpBackend.PYTORCH_NATIVE_SILU]

    op = registry.get_op("silu")

    assert isinstance(op, InstrumentedOp)
    # Still delegates correctly to the underlying implementation.
    x = torch.randn(2, 4)
    assert torch.allclose(op(x), torch.nn.functional.silu(x))


def test_kv_cache_fragmentation_recorded():
    pytest.importorskip("prometheus_client")

    inputs = StatelessForwardInputs(
        input_ids=torch.tensor([[0, 1, 2, 3, 0], [0, 2, 1, 4, 5]], dtype=torch.long),
        attention_mask=torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        ),
        completion_mask=torch.tensor(
            [[False, False, True, True, False], [False, True, True, False, False]]
        ),
    )
    config = PagedKVScoringConfig(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        block_size=2,
        kv_cache_dtype=torch.float32,
        kv_cache_blocks=8,
    )
    reservation = reserve_paged_kv_cache(inputs, config)
    assert reservation.required_blocks == 5
    assert reservation.reserved_blocks == 8

    collect_paged_kv_metrics(
        inputs,
        reservation,
        config=config,
        elapsed_seconds=0.0,
        use_cache_passed=True,
        cuda_tracking=False,
        model_kv_cache_summary=TensorTreeSummary(tensor_count=0, total_bytes=0),
    )

    value = _sample_value(
        metrics_module.KV_CACHE_FRAGMENTATION_RATIO,
        "rlkernel_kv_cache_fragmentation_ratio",
        baseline_kind="generation_engine_paged_kv_reservation",
    )
    assert value == pytest.approx(1.0 - 5 / 8)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_metrics_endpoint_serves_expected_names(monkeypatch):
    pytest.importorskip("prometheus_client")
    monkeypatch.setattr(metrics_module, "_server_started", False)

    port = _free_port()
    bound_port = start_metrics_server(port=port)
    assert bound_port == port

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        assert response.status == 200
        body = response.read().decode("utf-8")

    for name in (
        "rlkernel_op_calls_total",
        "rlkernel_op_latency_seconds",
        "rlkernel_op_fallbacks_total",
        "rlkernel_kv_cache_fragmentation_ratio",
    ):
        assert name in body


def test_start_metrics_server_returns_none_on_port_conflict(monkeypatch):
    pytest.importorskip("prometheus_client")
    monkeypatch.setattr(metrics_module, "_server_started", False)

    occupied_port = _free_port()
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", occupied_port))
    blocker.listen(1)
    try:
        # Must not raise OSError -- observability failures must never crash the caller.
        result = start_metrics_server(port=occupied_port)
        assert result is None
        # A failed bind must not mark the server as started, so a later call (e.g. on a
        # different port) can still succeed.
        assert metrics_module._server_started is False
    finally:
        blocker.close()


@pytest.mark.parametrize("raw_value", ["1yes", "enabled", "TRUE_ISH", "2"])
def test_env_flag_warns_and_falls_back_to_default_on_unrecognized_value(monkeypatch, raw_value):
    monkeypatch.setenv("RL_KERNEL_UNIT_TEST_BOOL_FLAG", raw_value)

    warnings = []
    monkeypatch.setattr(
        metrics_module.logger, "warn_once", lambda msg, *a: warnings.append(msg % a if a else msg)
    )

    assert metrics_module._env_flag("RL_KERNEL_UNIT_TEST_BOOL_FLAG", default=False) is False
    assert metrics_module._env_flag("RL_KERNEL_UNIT_TEST_BOOL_FLAG", default=True) is True
    assert len(warnings) == 2
    assert all(raw_value in w for w in warnings)


def test_registry_all_backends_fail_raises_and_records_every_fallback(monkeypatch):
    pytest.importorskip("prometheus_client")

    registry = KernelRegistry()
    op_type = "unit_test_all_fail_op_type"
    candidates = [OpBackend.PYTORCH_NATIVE, OpBackend.PYTORCH_NATIVE_SILU]
    for platform in ("cpu", "cuda", "rocm"):
        registry._priority_map[platform][op_type] = list(candidates)

    monkeypatch.setattr(KernelRegistry, "_load_backend", lambda self, backend: None)

    before = {
        backend.name: _sample_value(
            metrics_module.OP_FALLBACKS_TOTAL,
            "rlkernel_op_fallbacks_total",
            op_type=op_type,
            failed_backend=backend.name,
        )
        or 0.0
        for backend in candidates
    }

    with pytest.raises(RuntimeError, match=op_type):
        registry.get_op(op_type)

    for backend in candidates:
        after = _sample_value(
            metrics_module.OP_FALLBACKS_TOTAL,
            "rlkernel_op_fallbacks_total",
            op_type=op_type,
            failed_backend=backend.name,
        )
        assert after == before[backend.name] + 1


def test_instrumented_op_still_raises_and_records_metrics_on_failure():
    pytest.importorskip("prometheus_client")

    class _FailingOp:
        def __call__(self, *args, **kwargs):
            raise ValueError("boom")

    op_type, backend = "unit_test_failing_op", "UNIT_TEST_FAILING_BACKEND"
    wrapped = InstrumentedOp(_FailingOp(), op_type=op_type, backend=backend)

    before = (
        _sample_value(
            metrics_module.OP_CALLS_TOTAL,
            "rlkernel_op_calls_total",
            op_type=op_type,
            backend=backend,
            method="__call__",
        )
        or 0.0
    )

    with pytest.raises(ValueError, match="boom"):
        wrapped()

    after = _sample_value(
        metrics_module.OP_CALLS_TOTAL,
        "rlkernel_op_calls_total",
        op_type=op_type,
        backend=backend,
        method="__call__",
    )
    # The exception must propagate untouched (no swallowing), but the `finally` block should
    # still have recorded the call -- a failed invocation still consumed real time and is
    # still useful signal for throughput/error-rate dashboards.
    assert after == before + 1
