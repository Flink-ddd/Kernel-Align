# Observability

RL-Kernel ships a two-layer observability stack:

- **NVTX tracing (micro):** opt-in per-operator `rlk::<op>` ranges around every C++ binding plus
  Python stage ranges (`rlk::rollout.generate`, `rlk::score`, `rlk::train_step`, ...),
  visible on an [Nsight Systems](https://developer.nvidia.com/nsight-systems) timeline.
  Set `RL_KERNEL_NVTX=1` only for profiling runs; normal runs retain just a cached branch.
- **Prometheus metrics (macro):** a per-process `/metrics` endpoint with token throughput,
  rollout QPS, stage latencies, hardware fallback state, KV-cache fragmentation, peak GPU
  memory, and training loss — plus a ready-made Grafana dashboard.

## Installation

Prometheus support is an optional extra; the engine runs unchanged without it:

```bash
pip install -e ".[observability]"     # adds prometheus-client
```

Without `prometheus-client` installed, every metrics call degrades to a silent no-op.

## Prometheus metrics

### Starting the exporter

The exporter never starts as an import side effect. Either call it explicitly:

```python
from rl_engine.observability import start_metrics_server

port = start_metrics_server()          # default 0.0.0.0:8000; port=0 -> ephemeral
```

or opt in via environment variables when running the examples:

```bash
RL_KERNEL_METRICS=1 python examples/grpo_single_gpu.py --steps 3 --device cpu &
curl -s localhost:8000/metrics | grep rl_kernel_
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `RL_KERNEL_METRICS` | unset | `1` → examples/entrypoints start the exporter |
| `RL_KERNEL_METRICS_PORT` | `8000` | Exporter port; `0` = ephemeral |
| `RL_KERNEL_METRICS_ADDR` | `0.0.0.0` | Bind address override |

For multi-process (e.g. Ray) deployments each worker exposes its own endpoint; use
`RL_KERNEL_METRICS_PORT=0` to avoid port collisions and let your scrape topology discover
the workers.

### Metric schema

All metrics use the `rl_kernel_` prefix and Prometheus base units (seconds, bytes).

| Metric | Type | Labels |
| --- | --- | --- |
| `rl_kernel_build_info` | Gauge (=1) | `version`, `device_type`, `backend_version`, `ext_available` |
| `rl_kernel_stage_duration_seconds` | Histogram | `stage` |
| `rl_kernel_requests_total` | Counter | `stage`, `status` (`ok`/`error`) |
| `rl_kernel_output_tokens_total` | Counter | `stage` |
| `rl_kernel_selected_backend_info` | Gauge (=1) | `op_type`, `backend`, `is_fallback` |
| `rl_kernel_hardware_fallback_total` | Counter | `op_type`, `requested_backend`, `selected_backend`, `reason` |
| `rl_kernel_kv_cache_blocks` | Gauge | `kind` (`required`/`reserved`) |
| `rl_kernel_kv_cache_fragmentation_ratio` | Gauge | — |
| `rl_kernel_gpu_peak_memory_bytes` | Gauge | `stage`, `kind` (`allocated`/`reserved`) |
| `rl_kernel_training_loss` | Gauge | — |
| `rl_kernel_weight_version` | Gauge | `role` (`published`/`consumed`) |

Fallbacks are reported at two layers: `selected_backend_info` captures the *state* chosen
once per process at kernel dispatch time (`is_fallback="true"` means a lower-priority
backend was selected), while `hardware_fallback_total` counts *events* — backend
import/instantiation failures and per-call attention-backend fallbacks
(flash-attention → eager).

### Useful queries

```promql
rate(rl_kernel_output_tokens_total[1m])                          # token throughput
rate(rl_kernel_requests_total{stage="rollout_generate"}[1m])     # rollout QPS
histogram_quantile(0.95, rate(rl_kernel_stage_duration_seconds_bucket[5m]))
sum(rl_kernel_selected_backend_info{is_fallback="true"}) / sum(rl_kernel_selected_backend_info)
```

### Prometheus scrape config

```yaml
scrape_configs:
  - job_name: rl-kernel
    scrape_interval: 5s
    static_configs:
      - targets: ["localhost:8000"]
```

### Grafana dashboard

Import [`examples/grafana-dashboard.json`](https://github.com/RL-Align/RL-Kernel/blob/main/examples/grafana-dashboard.json)
(Grafana ≥ 10, *Dashboards → Import*) and select your Prometheus datasource when prompted —
the dashboard binds it through the `${DS_PROMETHEUS}` input, no hardcoded UID.

### Using the facade in your own code

```python
from rl_engine.observability import metrics

with metrics.stage_timer("my_stage"):        # duration histogram + ok/error counter
    run_stage()
metrics.add_output_tokens("my_stage", n_tokens)
```

Rules the built-in instrumentation follows (and yours should too):

- per step / per request granularity — never per token in hot loops;
- label values only from finite enums in code, never user data;
- exceptions propagate unchanged (recorded with `status="error"`);
- never re-query something the stage already computed — see below.

### Overhead

Metrics use exact 16-call batched updates for the module-level registry: counters and
histogram buckets remain exact, gauges retain the latest value, and pending updates flush
before each scrape. This moves Prometheus locks and repeated fixed-label lookups off the
hot path. On an RTX 5090 with torch 2.8/CUDA 12.8, the complete default score path measured
below 1% overhead at the benchmark's toy shape (paired, randomly interleaved calls).

| Workload | Baseline stage time | With metrics | Overhead |
| --- | --- | --- | --- |
| `benchmark_stateless_executor` defaults (toy model, hidden 256 / vocab 4096) | 1.038 ms | 1.047 ms | +0.90 % |

With `RL_KERNEL_NVTX=1`, Python and C++ ranges intentionally add profiler annotation cost;
do not use that mode for the normal metrics-overhead acceptance run.

If you add instrumentation, be careful what you *read* to populate it. Querying
`torch.cuda.max_memory_allocated()` / `max_memory_reserved()` costs ~85 µs for the pair —
each one rebuilds the full `torch.cuda.memory_stats()` dict. Prefer values the stage has
already computed, or read `memory_stats()` once and pull both peaks out of it.

## NVTX tracing with Nsight Systems

Capture a timeline on a CUDA machine:

```bash
RL_KERNEL_NVTX=1 nsys profile -t nvtx,cuda -o rlk_trace \
  python examples/grpo_single_gpu.py --steps 3
nsys stats --report nvtx_sum rlk_trace.nsys-rep      # lists rlk::* ranges
```

On the timeline the Python stage ranges (`rlk::train_step` with `rlk::train.forward` /
`rlk::train.backward` / `rlk::train.optim_step` inside) contain the per-operator
`rlk::<binding_name>` ranges emitted by the C++ extension, correlated with kernel
execution on the CUDA streams.

Notes:

- NVTX v3 is header-only and ships with the CUDA Toolkit; no extra link dependency.
- ROCm builds compile the tracing macros to no-ops.
- To compile the extension without NVTX, build with `KERNEL_ALIGN_DISABLE_NVTX=1`.
- Python stage ranges silently no-op when CUDA is unavailable.
