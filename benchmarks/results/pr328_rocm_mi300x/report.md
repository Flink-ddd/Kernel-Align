# PR #328 ROCm vocab-parallel logprob performance analysis

> Operator-only benchmark. No model checkpoint or serving engine was used.

## Environment

| Item | Value |
|---|---|
| architecture | gfx942:sramecc+:xnack- |
| extension_symbols | hip_deterministic_logp_tile_stats, hip_deterministic_logp_backward |
| git_commit | e9f1d2a5c67a283bd987978614b4444a9416c1f0 |
| gpu | AMD Instinct MI300X |
| gpu_count | 8 |
| hip | 7.14.60850 |
| native_collective | torch.distributed ProcessGroupNCCL (RCCL on ROCm) |
| python | 3.12.3 |
| torch | 2.12.0+rocm7.14.0a20260608 |

## Methodology

- Qwen3 vocabulary `V=151936` split into 64 tiles of 2374 columns; seeded logits (`randn * 2.0`), random targets, every seventh token inactive.
- Measured paths:
  - `native`: `pytorch-vocab-parallel-logp-ws2`, the WS2 vocab-parallel reference operator: a PyTorch tile loop for the per-tile FP32 `(max, sumexp)` partials, all-gather of the partials, fixed global tile-order merge, and a PyTorch autograd backward.
  - `triton`: `triton-vocab-parallel-logp-ws2`, the same contract, transport, and merge with two Triton kernels (tile statistics read from the stored shard, fused backward); one source for CUDA and ROCm.
  - `hip`: `rocm-vocab-parallel-logp-ws2`, the same contract, transport, and merge with two HIP kernels: `hip_deterministic_logp_tile_stats` reads the stored BF16/FP16/FP32 shard directly (8-element vector loads, no FP32 copy) and `hip_deterministic_logp_backward` produces the gradient in one fused pass.
- Distributed: one process per GPU; the WS2 operators all-gather per-tile `(max, sumexp)` partials over RCCL and merge them in fixed global tile order; CP ranks shard tokens and never enter the merge.
- Forward returns the selected-token logprob and the vocabulary LSE; forward+backward computes `grad_logits` for `sum(active * logp)`. The WS2 operators run with `validate=False`; the `validate=True` production entry point is measured separately.
- Single-GPU timing: GPU events, median and p95. Distributed timing: synchronized wall clock, slowest rank per sample. Peak memory is the per-call increase in `torch.cuda.max_memory_allocated` (distributed: max over ranks).
- Accuracy is against an FP64 `logsumexp` of the same (BF16-rounded) logits. Repeat = two identical calls are bitwise equal; batch-invariant = a row computed alone is bitwise equal to the same row inside the batch; TP-replicated = every TP rank holds identical bits.
- 5 warmups, 20 measured forward samples, 10 measured forward+backward samples. Raw medians, p95, minimum, maximum, and every measured path are in `results.json`.
- Tables show 2048 tokens; the figures cover the full token sweep.

Reproduce this report from the repository root:

```bash
python benchmarks/benchmark_rocm_logp.py \
  --warmup 5 \
  --samples 20 \
  --training-samples 10 \
  --report-paths ws2-reference,ws2-triton,ws2-rocm \
  --rename ws2-reference=native,ws2-triton=triton,ws2-rocm=hip \
  --report-baseline ws2-reference \
  --table-tokens 2048 \
  --output-dir benchmarks/results/pr328_rocm_mi300x
```

## Key findings

- Single GPU: `triton` is 5.26-6.45x faster than `native` in forward and 4.29-5.71x in forward+backward, with 0.15-0.33x its peak memory.
- Distributed: `triton` is 1.62-4.28x faster than `native` in forward and 1.82-4.05x in forward+backward across 6 TP/CP topologies, at 0.15x the per-rank peak memory (absolute forward 0.824-1.115 ms).
- Single GPU: `hip` is 5.59-6.88x faster than `native` in forward and 4.47-6.07x in forward+backward, with 0.15-0.33x its peak memory.
- Distributed: `hip` is 1.70-4.67x faster than `native` in forward and 2.05-4.49x in forward+backward across 6 TP/CP topologies, at 0.15x the per-rank peak memory (absolute forward 0.755-1.063 ms).
- The `hip_deterministic_logp_tile_stats` kernel alone is 9.6-9.9x faster than the PyTorch tile loop and allocates 57x less transient memory (it writes only the `[tokens, 64]` FP32 partials).
- `triton` vs `native`: tile maxima are bitwise equal; sumexp partials differ only by FP32 summation order, so final outputs differ in 0-169 elements per case with relative-L2 0.0e+00-6.9e-08. Both paths are equally close to FP64.
- `hip` vs `native`: tile maxima are bitwise equal; sumexp partials differ only by FP32 summation order, so final outputs differ in 0-65 elements per case with relative-L2 0.0e+00-1.3e-08. Both paths are equally close to FP64.
- Repeat bitwise: yes; batch-invariant: yes; all gradients finite: yes.
- Distributed: TP-replicated and repeat bitwise on every topology: yes.

## Single-GPU logprob (BF16 logits, V=151936)

### Forward

| Tokens | Path | Median (ms) | p95 (ms) | Speedup vs native | Peak MiB | logp max-abs vs FP64 | LSE max-abs vs FP64 | Repeat | Batch-inv |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 2048 | native | 6.1387 | 6.3072 | 1.00× | 1245.0 | 1.557e-06 | 1.382e-06 | yes | yes |
| 2048 | triton | 0.9514 | 0.9801 | 6.45× | 2.0 | 1.557e-06 | 1.318e-06 | yes | yes |
| 2048 | hip | 0.8923 | 0.9162 | 6.88× | 2.0 | 1.621e-06 | 1.318e-06 | yes | yes |

### Forward+backward

| Tokens | Path | Median (ms) | p95 (ms) | Speedup vs native | Peak MiB | Memory vs native | Grad finite |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 2048 | native | 12.3477 | 12.4680 | 1.00× | 7715.7 | 1.00× | yes |
| 2048 | triton | 2.1619 | 2.1945 | 5.71× | 1187.1 | 0.15× | yes |
| 2048 | hip | 2.0359 | 2.0513 | 6.07× | 1187.1 | 0.15× | yes |

### Numerics versus `native`

| Tokens | Path | Mismatched elements (logp+LSE) | Relative L2 |
|---:|---|---:|---:|
| 2048 | triton | 169 | 1.713e-08 |
| 2048 | hip | 65 | 1.098e-08 |

## Single-GPU logprob (FP32 logits, V=151936)

### Forward

| Tokens | Path | Median (ms) | p95 (ms) | Speedup vs native | Peak MiB | logp max-abs vs FP64 | LSE max-abs vs FP64 | Repeat | Batch-inv |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 2048 | native | 5.5791 | 5.6527 | 1.00× | 58.0 | 1.747e-06 | 1.271e-06 | yes | yes |
| 2048 | triton | 1.0602 | 1.0851 | 5.26× | 2.0 | 1.711e-06 | 1.197e-06 | yes | yes |
| 2048 | hip | 0.9972 | 1.0351 | 5.59× | 2.0 | 1.747e-06 | 1.271e-06 | yes | yes |

### Forward+backward

| Tokens | Path | Median (ms) | p95 (ms) | Speedup vs native | Peak MiB | Memory vs native | Grad finite |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 2048 | native | 12.1054 | 12.1808 | 1.00× | 7122.2 | 1.00× | yes |
| 2048 | triton | 2.8249 | 2.8799 | 4.29× | 2374.1 | 0.33× | yes |
| 2048 | hip | 2.7086 | 3.3877 | 4.47× | 2374.1 | 0.33× | yes |

### Numerics versus `native`

| Tokens | Path | Mismatched elements (logp+LSE) | Relative L2 |
|---:|---|---:|---:|
| 2048 | triton | 144 | 1.466e-08 |
| 2048 | hip | 37 | 8.004e-09 |

### `validate=True` production entry point (hip, BF16)

| Tokens | validate=False (ms) | validate=True (ms) | Overhead |
|---:|---:|---:|---:|
| 2048 | 0.9116 | 1.0558 | 1.16× |

`validate=True` adds host-side target-range checks and a non-finite LSE check that synchronizes the stream; the cost is a fixed per-call overhead.

## Tile-stats kernel

`hip_deterministic_logp_tile_stats` computes the per-row, per-tile FP32 `(max, sumexp)` partials that the operator all-gathers and merges; the PyTorch tile loop is what `native` uses for the same step. Tile maxima are bitwise equal; sums differ only by FP32 summation order.

| Logits dtype | Tokens | PyTorch tile loop (ms) | HIP kernel on FP32 (ms) | HIP kernel on stored dtype (ms) | Speedup | Loop peak MiB | HIP peak MiB | Max bitwise | sumexp max rel | Repeat |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|
| bf16 | 2048 | 5.0990 | 0.5171 | 0.4669 | 9.86× | 56.6 | 1.0 | yes | 3.64e-07 | yes |
| fp32 | 2048 | 5.1025 | 0.5328 | 0.5331 | 9.58× | 56.6 | 1.0 | yes | 4.48e-07 | yes |

## Distributed vocab-parallel logprob (BF16, RCCL)

### Forward

| Topology | Tokens | Path | Median (ms) | p95 (ms) | Speedup vs native | Peak MiB/rank | logp max-abs vs FP64 | TP-replicated | Repeat |
|---|---:|---|---:|---:|---:|---:|---:|:---:|:---:|
| tp2 | 2048 | native | 3.5877 | 3.6322 | 1.00× | 650.3 | 1.452e-06 | yes | yes |
| tp2 | 2048 | triton | 0.8894 | 0.9138 | 4.03× | 3.0 | 1.452e-06 | yes | yes |
| tp2 | 2048 | hip | 0.8487 | 0.8818 | 4.23× | 3.0 | 1.452e-06 | yes | yes |
| tp4 | 2048 | native | 2.2691 | 2.3080 | 1.00× | 353.0 | 1.452e-06 | yes | yes |
| tp4 | 2048 | triton | 0.9221 | 0.9652 | 2.46× | 2.5 | 1.452e-06 | yes | yes |
| tp4 | 2048 | hip | 0.8714 | 0.9162 | 2.60× | 2.5 | 1.452e-06 | yes | yes |
| tp8 | 2048 | native | 1.8066 | 1.8356 | 1.00× | 204.3 | 1.452e-06 | yes | yes |
| tp8 | 2048 | triton | 1.1149 | 1.1961 | 1.62× | 2.3 | 1.452e-06 | yes | yes |
| tp8 | 2048 | hip | 1.0631 | 1.1280 | 1.70× | 2.3 | 1.452e-06 | yes | yes |
| tp2_cp2 | 2048 | native | 3.3882 | 3.5075 | 1.00× | 325.5 | 1.452e-06 | yes | yes |
| tp2_cp2 | 2048 | triton | 0.8308 | 0.9605 | 4.08× | 1.5 | 1.452e-06 | yes | yes |
| tp2_cp2 | 2048 | hip | 0.7809 | 0.8174 | 4.34× | 1.5 | 1.452e-06 | yes | yes |
| tp4_cp2 | 2048 | native | 2.2382 | 2.2630 | 1.00× | 176.7 | 1.452e-06 | yes | yes |
| tp4_cp2 | 2048 | triton | 0.8831 | 0.9622 | 2.53× | 1.3 | 1.452e-06 | yes | yes |
| tp4_cp2 | 2048 | hip | 0.8386 | 0.8978 | 2.67× | 1.3 | 1.452e-06 | yes | yes |
| tp2_cp4 | 2048 | native | 3.5261 | 3.6071 | 1.00× | 163.1 | 1.452e-06 | yes | yes |
| tp2_cp4 | 2048 | triton | 0.8240 | 1.3511 | 4.28× | 0.8 | 1.452e-06 | yes | yes |
| tp2_cp4 | 2048 | hip | 0.7547 | 0.7834 | 4.67× | 0.8 | 1.452e-06 | yes | yes |

### Forward+backward

| Topology | Tokens | Path | Median (ms) | p95 (ms) | Speedup vs native | Peak MiB/rank | Memory vs native | Grad finite |
|---|---:|---|---:|---:|---:|---:|---:|:---:|
| tp2 | 2048 | native | 7.1482 | 7.5344 | 1.00× | 3857.9 | 1.00× | yes |
| tp2 | 2048 | triton | 1.7667 | 1.8881 | 4.05× | 593.6 | 0.15× | yes |
| tp2 | 2048 | hip | 1.5938 | 1.7476 | 4.49× | 593.6 | 0.15× | yes |
| tp4 | 2048 | native | 4.4688 | 4.5386 | 1.00× | 1929.0 | 1.00× | yes |
| tp4 | 2048 | triton | 1.7425 | 2.1374 | 2.56× | 296.8 | 0.15× | yes |
| tp4 | 2048 | hip | 1.5001 | 1.6030 | 2.98× | 296.8 | 0.15× | yes |
| tp8 | 2048 | native | 3.3579 | 3.9040 | 1.00× | 964.5 | 1.00× | yes |
| tp8 | 2048 | triton | 1.8465 | 1.9464 | 1.82× | 148.5 | 0.15× | yes |
| tp8 | 2048 | hip | 1.6417 | 1.8066 | 2.05× | 148.5 | 0.15× | yes |
| tp2_cp2 | 2048 | native | 5.6092 | 5.7342 | 1.00× | 1929.0 | 1.00× | yes |
| tp2_cp2 | 2048 | triton | 1.4580 | 1.7000 | 3.85× | 296.8 | 0.15× | yes |
| tp2_cp2 | 2048 | hip | 1.3925 | 1.6368 | 4.03× | 296.8 | 0.15× | yes |
| tp4_cp2 | 2048 | native | 3.7864 | 3.8899 | 1.00× | 964.5 | 1.00× | yes |
| tp4_cp2 | 2048 | triton | 1.6803 | 1.7273 | 2.25× | 148.4 | 0.15× | yes |
| tp4_cp2 | 2048 | hip | 1.4029 | 1.4394 | 2.70× | 148.4 | 0.15× | yes |
| tp2_cp4 | 2048 | native | 4.9700 | 5.0383 | 1.00× | 964.5 | 1.00× | yes |
| tp2_cp4 | 2048 | triton | 1.5611 | 1.6497 | 3.18× | 148.4 | 0.15× | yes |
| tp2_cp4 | 2048 | hip | 1.3959 | 1.5368 | 3.56× | 148.4 | 0.15× | yes |

### Numerics versus `native` (distributed)

| Topology | Tokens | Path | Mismatched elements (logp+LSE) | Relative L2 |
|---|---:|---|---:|---:|
| tp2 | 2048 | triton | 137 | 1.454e-08 |
| tp2 | 2048 | hip | 54 | 9.958e-09 |
| tp4 | 2048 | triton | 137 | 1.454e-08 |
| tp4 | 2048 | hip | 54 | 9.958e-09 |
| tp8 | 2048 | triton | 137 | 1.454e-08 |
| tp8 | 2048 | hip | 54 | 9.958e-09 |
| tp2_cp2 | 2048 | triton | 137 | 1.477e-08 |
| tp2_cp2 | 2048 | hip | 54 | 1.069e-08 |
| tp4_cp2 | 2048 | triton | 137 | 1.477e-08 |
| tp4_cp2 | 2048 | hip | 54 | 1.069e-08 |
| tp2_cp4 | 2048 | triton | 137 | 1.735e-08 |
| tp2_cp4 | 2048 | hip | 54 | 1.162e-08 |

## Figures

![Single-GPU latency](single_gpu_latency.png)

![Single-GPU peak memory](single_gpu_memory.png)

![Distributed latency](distributed_logp_latency.png)

