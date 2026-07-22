# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import json
import pathlib
import re
import sys
import urllib.request
from typing import Any, Optional

import pytest
import torch

from rl_engine.executors.paged_kv_baseline import PagedKVScoringConfig, reserve_paged_kv_cache
from rl_engine.executors.stateless_executor import (
    StatelessForwardConfig,
    StatelessForwardExecutor,
    StatelessForwardInputs,
)
from rl_engine.kernels.registry import KernelRegistry, OpBackend
from rl_engine.observability import SCHEMA_METRIC_NAMES, MetricsRegistry, metrics, nvtx_range
from rl_engine.platforms.device import device_ctx

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DASHBOARD_PATH = REPO_ROOT / "examples" / "grafana-dashboard.json"


def _sample_value(
    registry: MetricsRegistry,
    name: str,
    labels: Optional[dict[str, str]] = None,
) -> Optional[float]:
    collector_registry = registry.collector_registry
    assert collector_registry is not None
    return collector_registry.get_sample_value(name, labels or {})


def _make_score_inputs(batch_size: int = 2, seq_len: int = 6) -> StatelessForwardInputs:
    torch.manual_seed(0)
    input_ids = torch.randint(0, 16, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    completion_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    completion_mask[:, 0] = False
    return StatelessForwardInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
    )


class _TinyLogitsModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache=None):
        del attention_mask, use_cache
        return {"logits": self.proj(self.embedding(input_ids))}


class _FlashFailingModel(_TinyLogitsModel):
    """Fails the first forward like a missing flash-attention install."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache=None):
        self.calls += 1
        if self.calls == 1:
            raise ImportError("flash attention kernels are unavailable")
        return super().forward(input_ids, attention_mask, use_cache)


class _AlwaysFailingModel(_TinyLogitsModel):
    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache=None):
        raise RuntimeError("score exploded")


@pytest.fixture()
def blocked_prometheus(monkeypatch):
    """Make `import prometheus_client` fail inside the fixture scope."""
    monkeypatch.setitem(sys.modules, "prometheus_client", None)


@pytest.fixture()
def fresh_metrics():
    pytest.importorskip("prometheus_client")
    return MetricsRegistry()


# ---------------------------------------------------------------------------
# 1. No-op degradation without prometheus_client
# ---------------------------------------------------------------------------


def test_facade_is_noop_without_prometheus(blocked_prometheus):
    registry = MetricsRegistry()

    assert registry.enabled is False
    assert registry.collector_registry is None

    with registry.stage_timer("score"):
        pass
    registry.add_output_tokens("score", 3)
    registry.set_selected_backend("logp", "PYTORCH_NATIVE", is_fallback=False)
    registry.record_hardware_fallback(
        "logp",
        requested_backend="a",
        selected_backend="b",
        reason="import_error",
    )
    registry.set_kv_cache_blocks(required=1, reserved=2)
    registry.set_gpu_peak_memory("score", allocated_bytes=1, reserved_bytes=2)
    registry.set_training_loss(0.5)
    registry.set_weight_version("published", 1)


def test_stage_timer_reraises_without_prometheus(blocked_prometheus):
    registry = MetricsRegistry()
    with pytest.raises(ValueError, match="boom"):
        with registry.stage_timer("score"):
            raise ValueError("boom")


def test_executor_runs_unchanged_without_prometheus(blocked_prometheus, monkeypatch):
    disabled = MetricsRegistry()
    monkeypatch.setattr("rl_engine.executors.stateless_executor.obs_metrics", disabled)

    executor = StatelessForwardExecutor(
        _TinyLogitsModel(),
        StatelessForwardConfig(mode="reference", attention_backend="model_default"),
    )
    result = executor.score(_make_score_inputs())

    assert result.reference_logps is not None
    assert result.metrics["active_completion_tokens"] == 10


# ---------------------------------------------------------------------------
# 2. /metrics endpoint
# ---------------------------------------------------------------------------


def test_metrics_endpoint_serves_schema_families():
    pytest.importorskip("prometheus_client")
    from rl_engine.observability.server import start_metrics_server

    # Touch a couple of facade methods so families beyond build_info have data.
    metrics.set_training_loss(1.25)
    with metrics.stage_timer("score"):
        pass

    port = start_metrics_server(port=0, addr="127.0.0.1")
    assert port > 0
    # Idempotent second call reuses the running exporter.
    assert start_metrics_server(port=0, addr="127.0.0.1") == port

    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    assert "rl_kernel_build_info" in body
    assert "rl_kernel_training_loss" in body
    assert "rl_kernel_stage_duration_seconds" in body
    assert "rl_kernel_requests_total" in body


def test_env_helper_disabled_by_default(monkeypatch):
    from rl_engine.observability.server import maybe_start_metrics_server_from_env

    monkeypatch.delenv("RL_KERNEL_METRICS", raising=False)
    assert maybe_start_metrics_server_from_env() is None


# ---------------------------------------------------------------------------
# 3. Kernel registry dispatch + fallback instrumentation
# ---------------------------------------------------------------------------


def _active_platform() -> str:
    """Priority-map key `KernelRegistry.get_op` resolves to on this host."""
    if device_ctx.is_rocm:
        return "rocm"
    return "cuda" if device_ctx.device_type == "cuda" else "cpu"


def test_registry_records_selected_backend(fresh_metrics, monkeypatch):
    monkeypatch.setattr("rl_engine.kernels.registry.metrics", fresh_metrics)
    registry = KernelRegistry()
    # Pin a single known backend so the assertion holds on CPU and GPU hosts alike.
    monkeypatch.setitem(
        registry._priority_map[_active_platform()],
        "logp",
        [OpBackend.PYTORCH_NATIVE],
    )

    op = registry.get_op("logp")

    assert op is not None
    assert (
        _sample_value(
            fresh_metrics,
            "rl_kernel_selected_backend_info",
            {"op_type": "logp", "backend": "PYTORCH_NATIVE", "is_fallback": "false"},
        )
        == 1
    )


def test_registry_records_hardware_fallback(fresh_metrics, monkeypatch):
    monkeypatch.setattr("rl_engine.kernels.registry.metrics", fresh_metrics)
    registry = KernelRegistry()
    monkeypatch.setitem(
        registry._priority_map[_active_platform()],
        "logp",
        [OpBackend.FLASH_ATTN, OpBackend.PYTORCH_NATIVE],
    )

    real_import_module = importlib.import_module

    def failing_import(path: str, *args: Any, **kwargs: Any):
        if "flash_attn" in path:
            raise ModuleNotFoundError("No module named 'flash_attn'", name="flash_attn")
        return real_import_module(path, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", failing_import)

    op = registry.get_op("logp")

    assert op is not None
    assert (
        _sample_value(
            fresh_metrics,
            "rl_kernel_hardware_fallback_total",
            {
                "op_type": "logp",
                "requested_backend": "FLASH_ATTN",
                "selected_backend": "PYTORCH_NATIVE",
                "reason": "import_error",
            },
        )
        == 1
    )
    assert (
        _sample_value(
            fresh_metrics,
            "rl_kernel_selected_backend_info",
            {"op_type": "logp", "backend": "PYTORCH_NATIVE", "is_fallback": "true"},
        )
        == 1
    )


# ---------------------------------------------------------------------------
# 4. Timer semantics
# ---------------------------------------------------------------------------


def test_stage_timer_records_ok_and_error(fresh_metrics):
    with fresh_metrics.stage_timer("train_step"):
        pass
    with pytest.raises(RuntimeError, match="exploded"):
        with fresh_metrics.stage_timer("train_step"):
            raise RuntimeError("exploded")

    ok = _sample_value(
        fresh_metrics,
        "rl_kernel_requests_total",
        {"stage": "train_step", "status": "ok"},
    )
    error = _sample_value(
        fresh_metrics,
        "rl_kernel_requests_total",
        {"stage": "train_step", "status": "error"},
    )
    duration_count = _sample_value(
        fresh_metrics,
        "rl_kernel_stage_duration_seconds_count",
        {"stage": "train_step"},
    )
    assert ok == 1
    assert error == 1
    assert duration_count == 2


def test_batched_stage_updates_flush_exact_values_on_collection():
    registry = MetricsRegistry(stage_batch_size=8)
    registry.record_stage(
        "score",
        0.002,
        status="ok",
        output_tokens=3,
        allocated_bytes=10,
        reserved_bytes=20,
    )
    registry.record_stage(
        "score",
        0.020,
        status="ok",
        output_tokens=4,
        allocated_bytes=30,
        reserved_bytes=40,
    )

    assert (
        _sample_value(
            registry,
            "rl_kernel_requests_total",
            {"stage": "score", "status": "ok"},
        )
        == 2
    )
    assert (
        _sample_value(
            registry,
            "rl_kernel_stage_duration_seconds_count",
            {"stage": "score"},
        )
        == 2
    )
    assert _sample_value(
        registry,
        "rl_kernel_stage_duration_seconds_sum",
        {"stage": "score"},
    ) == pytest.approx(0.022)
    assert (
        _sample_value(
            registry,
            "rl_kernel_output_tokens_total",
            {"stage": "score"},
        )
        == 7
    )
    assert (
        _sample_value(
            registry,
            "rl_kernel_gpu_peak_memory_bytes",
            {"stage": "score", "kind": "allocated"},
        )
        == 30
    )


def test_metrics_registry_rejects_invalid_stage_batch_size():
    with pytest.raises(ValueError, match="stage_batch_size"):
        MetricsRegistry(stage_batch_size=0)


# ---------------------------------------------------------------------------
# 5. Executor instrumentation
# ---------------------------------------------------------------------------


def test_score_records_stage_and_tokens(fresh_metrics, monkeypatch):
    monkeypatch.setattr("rl_engine.executors.stateless_executor.obs_metrics", fresh_metrics)

    executor = StatelessForwardExecutor(
        _TinyLogitsModel(),
        StatelessForwardConfig(mode="reference", attention_backend="model_default"),
    )
    result = executor.score(_make_score_inputs())

    assert result.metrics["active_completion_tokens"] == 10
    assert (
        _sample_value(
            fresh_metrics,
            "rl_kernel_requests_total",
            {"stage": "score", "status": "ok"},
        )
        == 1
    )
    assert _sample_value(fresh_metrics, "rl_kernel_output_tokens_total", {"stage": "score"}) == 10


def test_score_records_attention_fallback_event(fresh_metrics, monkeypatch):
    monkeypatch.setattr("rl_engine.executors.stateless_executor.obs_metrics", fresh_metrics)

    executor = StatelessForwardExecutor(
        _FlashFailingModel(),
        StatelessForwardConfig(mode="reference", attention_backend="flash_attention_2"),
    )
    result = executor.score(_make_score_inputs())

    assert result.metrics["attention_backend_fallback"] is True
    assert (
        _sample_value(
            fresh_metrics,
            "rl_kernel_hardware_fallback_total",
            {
                "op_type": "attention_backend",
                "requested_backend": "flash_attention_2",
                "selected_backend": "eager",
                "reason": "ImportError",
            },
        )
        == 1
    )


def test_score_records_error_without_changing_exception(fresh_metrics, monkeypatch):
    monkeypatch.setattr("rl_engine.executors.stateless_executor.obs_metrics", fresh_metrics)
    executor = StatelessForwardExecutor(
        _AlwaysFailingModel(),
        StatelessForwardConfig(mode="reference", attention_backend="model_default"),
    )

    with pytest.raises(RuntimeError, match="score exploded"):
        executor.score(_make_score_inputs())

    assert (
        _sample_value(
            fresh_metrics,
            "rl_kernel_requests_total",
            {"stage": "score", "status": "error"},
        )
        == 1
    )


def test_score_records_error_on_input_validation_failure(fresh_metrics, monkeypatch):
    # Regression: a shape/device mismatch is rejected before the forward pass,
    # so it must still be recorded as an errored stage rather than escaping
    # uninstrumented.
    monkeypatch.setattr("rl_engine.executors.stateless_executor.obs_metrics", fresh_metrics)
    executor = StatelessForwardExecutor(
        _TinyLogitsModel(),
        StatelessForwardConfig(mode="reference", attention_backend="model_default"),
    )
    good = _make_score_inputs(batch_size=2, seq_len=6)
    mismatched = StatelessForwardInputs(
        input_ids=good.input_ids,
        attention_mask=torch.ones(2, 7, dtype=torch.long),  # wrong shape
        completion_mask=good.completion_mask,
    )

    with pytest.raises(ValueError, match="attention_mask shape"):
        executor.score(mismatched)

    assert (
        _sample_value(
            fresh_metrics,
            "rl_kernel_requests_total",
            {"stage": "score", "status": "error"},
        )
        == 1
    )


def test_paged_kv_gauges(fresh_metrics, monkeypatch):
    monkeypatch.setattr("rl_engine.executors.paged_kv_baseline.obs_metrics", fresh_metrics)
    from rl_engine.executors.paged_kv_baseline import collect_paged_kv_metrics
    from rl_engine.executors.stateless_executor import summarize_tensor_tree

    inputs = _make_score_inputs(batch_size=2, seq_len=6)
    config = PagedKVScoringConfig(block_size=4, kv_cache_blocks=8)
    reservation = reserve_paged_kv_cache(inputs, config)
    collect_paged_kv_metrics(
        inputs,
        reservation,
        config=config,
        elapsed_seconds=0.1,
        use_cache_passed=True,
        cuda_tracking=False,
        model_kv_cache_summary=summarize_tensor_tree(None),
    )

    # 2 sequences x 6 tokens with block_size=4 -> 2 blocks each.
    assert _sample_value(fresh_metrics, "rl_kernel_kv_cache_blocks", {"kind": "required"}) == 4
    assert _sample_value(fresh_metrics, "rl_kernel_kv_cache_blocks", {"kind": "reserved"}) == 8
    assert _sample_value(fresh_metrics, "rl_kernel_kv_cache_fragmentation_ratio") == 0.5


def test_training_observability_helper(fresh_metrics, monkeypatch):
    monkeypatch.setattr("rl_engine.executors.deepspeed_trainer.obs_metrics", fresh_metrics)
    from rl_engine.executors.deepspeed_trainer import _record_training_observability

    _record_training_observability(
        {"loss": 0.75, "active_tokens": 12},
        device=torch.device("cpu"),
    )

    assert _sample_value(fresh_metrics, "rl_kernel_training_loss") == 0.75
    assert (
        _sample_value(fresh_metrics, "rl_kernel_output_tokens_total", {"stage": "train_step"}) == 12
    )


def test_nvtx_range_noops_without_cuda():
    with nvtx_range("rlk::test"):
        value = 41 + 1
    assert value == 42


def test_nvtx_requires_explicit_runtime_opt_in(monkeypatch):
    nvtx_module = importlib.import_module("rl_engine.observability.nvtx")
    monkeypatch.delenv(nvtx_module.RL_KERNEL_NVTX, raising=False)
    nvtx_module._nvtx_enabled.cache_clear()
    nvtx_module._nvtx_module.cache_clear()
    assert nvtx_module._nvtx_module() is None

    monkeypatch.setenv(nvtx_module.RL_KERNEL_NVTX, "1")
    nvtx_module._nvtx_enabled.cache_clear()
    assert nvtx_module._nvtx_enabled() is True
    nvtx_module._nvtx_enabled.cache_clear()
    nvtx_module._nvtx_module.cache_clear()


# ---------------------------------------------------------------------------
# 6. Dashboard contract
# ---------------------------------------------------------------------------


def test_dashboard_references_only_schema_metrics():
    dashboard = json.loads(DASHBOARD_PATH.read_text())

    exprs = [
        target["expr"]
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if "expr" in target
    ]
    assert exprs, "dashboard defines no queries"

    referenced = set()
    for expr in exprs:
        referenced.update(re.findall(r"rl_kernel_[a-z0-9_]+", expr))
    assert referenced, "dashboard queries reference no rl_kernel_ metrics"

    for name in referenced:
        base = re.sub(r"_(bucket|sum|count)$", "", name)
        assert base in SCHEMA_METRIC_NAMES, f"dashboard references unknown metric {name}"


def test_dashboard_has_no_hardcoded_datasource_uid():
    dashboard = json.loads(DASHBOARD_PATH.read_text())
    text = DASHBOARD_PATH.read_text()
    assert "${DS_PROMETHEUS}" in text
    for panel in dashboard["panels"]:
        uid = panel.get("datasource", {}).get("uid")
        assert uid == "${DS_PROMETHEUS}"


# ---------------------------------------------------------------------------
# 7. Isolation between registries
# ---------------------------------------------------------------------------


def test_metrics_registries_are_isolated():
    pytest.importorskip("prometheus_client")
    first = MetricsRegistry()
    second = MetricsRegistry()

    first.set_training_loss(1.0)
    second.set_training_loss(2.0)

    assert first.collector_registry is not second.collector_registry
    assert _sample_value(first, "rl_kernel_training_loss") == 1.0
    assert _sample_value(second, "rl_kernel_training_loss") == 2.0
