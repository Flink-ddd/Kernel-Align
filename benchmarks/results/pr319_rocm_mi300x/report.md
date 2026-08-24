# PR #319 — strict ROCm attention, MI300X operator benchmark

Operator-only: seeded Q/K/V, no checkpoint, tokenizer, or serving engine.

| Field | Value |
|---|---|
| `gpu` | AMD Instinct MI300X |
| `arch` | gfx942:sramecc+:xnack- |
| `device_count` | 8 |
| `torch` | 2.12.0+rocm7.14.0a20260608 |
| `hip` | 7.14.60850 |
| `dtype` | bf16 |
| `q_heads` | 32 |
| `kv_heads` | 8 |
| `head_dim` | 128 |

Backends: `native` = PyTorch SDPA with KV heads expanded; `triton` = flash-attn ROCm
Triton backend; `strict` = `aiter.rocm.ck_dense_mha` through WS2 contract dispatch.

## Latency and peak memory (BF16, causal prefill)

| B | S | Backend | Fwd median (ms) | Fwd p95 (ms) | Fwd peak MiB | Fwd+bwd median (ms) | Fwd+bwd peak MiB |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 1024 | native | 0.1438 | 0.1668 | 36.1 | 0.4741 | 84.3 |
| 1 | 1024 | triton | 0.3069 | 0.3300 | 32.1 | 1.7754 | 52.2 |
| 1 | 1024 | strict | 0.4045 | 0.4261 | 40.1 | 1.0480 | 1108.5 |
| 1 | 2048 | native | 0.2963 | 0.3018 | 72.3 | 1.0880 | 168.8 |
| 1 | 2048 | triton | 0.6404 | 0.6551 | 64.2 | 2.2759 | 104.5 |
| 1 | 2048 | strict | 0.4327 | 0.4475 | 80.3 | 2.0983 | 4265.0 |
| 1 | 4096 | native | 0.7014 | 0.7219 | 144.5 | 3.2456 | 337.5 |
| 1 | 4096 | triton | 1.4337 | 1.4886 | 128.5 | 4.6235 | 209.0 |
| 1 | 4096 | strict | 0.7415 | 0.7529 | 160.5 | 5.8933 | 16722.0 |
| 2 | 2048 | native | 0.4437 | 0.4696 | 144.5 | 1.9135 | 337.5 |
| 2 | 2048 | triton | 0.9821 | 0.9917 | 128.5 | 2.6656 | 209.0 |
| 2 | 2048 | strict | 0.7912 | 0.8101 | 120.5 | 3.9666 | 4369.3 |
| 4 | 2048 | native | 0.8211 | 0.8666 | 289.0 | 3.5754 | 675.0 |
| 4 | 2048 | triton | 1.7571 | 1.7957 | 257.0 | 5.1310 | 418.0 |
| 4 | 2048 | strict | 1.3856 | 1.4119 | 226.1 | 7.7189 | 4610.3 |

## Cost of the deterministic backward

Raw AITER `mha_bwd`, B=1, toggling only the `deterministic` flag. Peak values include the
already-resident forward tensors, so read the absolute det=on column for scaling.

| S | det=on median (ms) | det=off median (ms) | Time | det=on peak MiB | det=off peak MiB |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.4455 | 0.1917 | 2.32x | 1188.3 | 180.3 |
| 2048 | 1.3399 | 0.5687 | 2.36x | 4328.5 | 264.5 |
| 4096 | 5.0313 | 2.0607 | 2.44x | 16753.0 | 433.0 |

Deterministic backward peak memory scales as O(S^2): each doubling of S quadruples it.

## Batch-composition invariance of raw AITER

Batch vs the same rows submitted one at a time. Shape-dependent, which is why the strict
core materializes one logical row per call.

| B | S | out max abs | lse max abs | Invariant |
|---:|---:|---:|---:|---|
| 2 | 128 | 0.000000e+00 | 0.000000e+00 | yes |
| 2 | 256 | 0.000000e+00 | 0.000000e+00 | yes |
| 2 | 512 | 1.562500e-02 | 9.536743e-07 | **no** |
| 2 | 1024 | 0.000000e+00 | 0.000000e+00 | yes |
| 2 | 2048 | 0.000000e+00 | 0.000000e+00 | yes |
| 2 | 4096 | 0.000000e+00 | 0.000000e+00 | yes |
| 4 | 128 | 0.000000e+00 | 0.000000e+00 | yes |
| 4 | 256 | 7.812500e-03 | 9.536743e-07 | **no** |
| 4 | 512 | 1.562500e-02 | 9.536743e-07 | **no** |
| 4 | 1024 | 0.000000e+00 | 0.000000e+00 | yes |
| 4 | 2048 | 0.000000e+00 | 0.000000e+00 | yes |
| 4 | 4096 | 0.000000e+00 | 0.000000e+00 | yes |

Through the provider, every shape above is bitwise identical (max abs `0`).

## Reproduce

```bash
python benchmarks/benchmark_rocm_attention.py \
  --output benchmarks/results/pr319_rocm_mi300x/results.json
```
