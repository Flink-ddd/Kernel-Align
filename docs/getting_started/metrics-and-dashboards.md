# Metrics & Dashboards Guide

This guide explains how to expose RL-Kernel's live Prometheus `/metrics` endpoint and load the
sample Grafana dashboard, for cluster-level monitoring of kernel throughput, backend-fallback
rate, and KV-cache fragmentation across a training/rollout deployment.

For per-op kernel-launch tracing inside a single process (an `nsys` timeline), see the
[NVTX & Nsight Profiling Guide](nsys-profiling.md) instead — that is a micro-level, offline
trace; this page covers the macro-level, always-on metrics surface.

## 1. Install

Prometheus support is an optional dependency:

```bash
pip install -e .[observability]
```

Without it, every metrics function in `rl_engine.observability.metrics` degrades to a no-op and
logs a one-time warning — no other RL-Kernel functionality is affected.

## 2. Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `RL_KERNEL_ENABLE_OP_METRICS` | off | Opt-in: wrap `KernelRegistry.get_op(...)` results to record per-op call count and latency. Off by default because several tests assert on the concrete op class returned by `get_op(...)`. |
| `RL_KERNEL_ENABLE_METRICS_SERVER` | off | Opt-in: auto-start the `/metrics` HTTP endpoint from `RolloutExecutor` on kernel init. |
| `RL_KERNEL_METRICS_PORT` | `9400` | Base port for the `/metrics` endpoint. The actual bind port is `RL_KERNEL_METRICS_PORT + RANK` (falls back to `LOCAL_RANK`, then `0`), so multiple ranks on one node do not collide. |

Backend-fallback and KV-cache-fragmentation recording require no opt-in beyond having
`prometheus_client` installed — they never change any function's return type, so they are
always active once the dependency is present.

## 3. Start a Worker and Scrape It

```bash
RL_KERNEL_ENABLE_METRICS_SERVER=1 RL_KERNEL_ENABLE_OP_METRICS=1 \
  python examples/grpo_single_gpu.py --device cuda --steps 2 \
  --num-prompts 1 --samples-per-prompt 2 --prompt-len 2 --completion-len 3 \
  --vocab-size 16 --hidden-dim 8
```

In another shell:

```bash
curl http://localhost:9400/metrics
```

Confirm the response contains:

- `rlkernel_op_calls_total`
- `rlkernel_op_latency_seconds_bucket`
- `rlkernel_op_fallbacks_total`
- `rlkernel_kv_cache_fragmentation_ratio`

You can also start the server directly from Python without any environment variable, for
notebooks or ad hoc scripts:

```python
from rl_engine.observability.metrics import start_metrics_server

start_metrics_server(port=9400)
```

## 4. Point Prometheus at It

```yaml
scrape_configs:
  - job_name: rl-kernel
    static_configs:
      - targets: ["localhost:9400"]
```

For a multi-rank node, add one target per rank's resolved port
(`RL_KERNEL_METRICS_PORT + rank`).

## 5. Load the Sample Dashboard

Import `examples/grafana/rl_kernel_dashboard.json` into Grafana (**Dashboards → New → Import**),
and select your Prometheus datasource when prompted. It ships five panels:

- Scrape Target Up
- KV-Cache Fragmentation
- Op Throughput (calls/sec)
- Op Fallback Rate
- Op Latency p50 / p95 / p99

## Reporting Guidance

When sharing a dashboard screenshot or a metrics snapshot, include:

- The RL-Kernel commit and the exact command used to start the worker.
- Whether `RL_KERNEL_ENABLE_OP_METRICS` was set (call-count/latency panels are empty otherwise).
- The number of ranks/workers scraped and their resolved ports.

Keep committed docs focused on process and configuration. Point-in-time metrics snapshots and
dashboard screenshots should stay outside the repository.
