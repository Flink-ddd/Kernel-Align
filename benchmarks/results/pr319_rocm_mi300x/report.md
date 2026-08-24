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
| 1 | 1024 | native | 0.1449 | 0.1503 | 36.1 | 0.4758 | 84.3 |
| 1 | 1024 | triton | 0.3021 | 0.3315 | 32.1 | 1.2487 | 52.2 |
| 1 | 1024 | strict | 0.3980 | 0.4205 | 40.1 | 1.0305 | 1108.5 |
| 1 | 2048 | native | 0.2972 | 0.3010 | 72.3 | 1.0875 | 168.8 |
| 1 | 2048 | triton | 0.6379 | 0.6535 | 64.2 | 2.5059 | 104.5 |
| 1 | 2048 | strict | 0.4265 | 0.4365 | 80.3 | 2.1164 | 4265.0 |
| 1 | 4096 | native | 0.7020 | 0.7232 | 144.5 | 3.2428 | 337.5 |
| 1 | 4096 | triton | 1.4280 | 1.4877 | 128.5 | 4.6182 | 209.0 |
| 1 | 4096 | strict | 0.7343 | 0.7502 | 160.5 | 5.9652 | 16722.0 |
| 2 | 2048 | native | 0.4499 | 0.4715 | 144.5 | 1.9132 | 337.5 |
| 2 | 2048 | triton | 0.9764 | 0.9864 | 128.5 | 2.6696 | 209.0 |
| 2 | 2048 | strict | 0.7826 | 0.8113 | 120.5 | 3.9917 | 4369.3 |
| 4 | 2048 | native | 0.8225 | 0.8474 | 289.0 | 3.5825 | 675.0 |
| 4 | 2048 | triton | 1.7539 | 1.7674 | 257.0 | 5.1151 | 418.0 |
| 4 | 2048 | strict | 1.3809 | 1.4118 | 226.1 | 7.6638 | 4610.3 |

## Cost of the deterministic backward

Raw AITER `mha_bwd`, B=1, toggling only the `deterministic` flag. Peak values include the
already-resident forward tensors, so read the absolute det=on column for scaling.

| S | det=on median (ms) | det=off median (ms) | Time | det=on peak MiB | det=off peak MiB |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.4490 | 0.1903 | 2.36x | 1188.3 | 180.3 |
| 2048 | 1.3386 | 0.5687 | 2.35x | 4328.5 | 264.5 |
| 4096 | 5.0450 | 2.0415 | 2.47x | 16753.0 | 433.0 |

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

## TP-degree invariance of raw AITER

A head shard computed under TP=N vs the same slice of an unsharded run. TP performs no
cross-rank reduction in attention, so any nonzero value here means the kernel's result
depends on how many heads shared the launch - i.e. changing TP degree changes the numbers.

| S | TP | Local Hq | Local Hkv | out max abs | lse max abs | Invariant |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 2 | 16 | 4 | 0.000000e+00 | 0.000000e+00 | yes |
| 512 | 4 | 8 | 2 | 0.000000e+00 | 0.000000e+00 | yes |
| 512 | 8 | 4 | 1 | 0.000000e+00 | 0.000000e+00 | yes |
| 1024 | 2 | 16 | 4 | 3.906250e-03 | 1.430511e-06 | **no** |
| 1024 | 4 | 8 | 2 | 3.906250e-03 | 1.430511e-06 | **no** |
| 1024 | 8 | 4 | 1 | 3.906250e-03 | 1.430511e-06 | **no** |
| 2048 | 2 | 16 | 4 | 0.000000e+00 | 0.000000e+00 | yes |
| 2048 | 4 | 8 | 2 | 7.812500e-03 | 2.861023e-06 | **no** |
| 2048 | 8 | 4 | 1 | 7.812500e-03 | 2.861023e-06 | **no** |
| 4096 | 2 | 16 | 4 | 0.000000e+00 | 0.000000e+00 | yes |
| 4096 | 4 | 8 | 2 | 0.000000e+00 | 0.000000e+00 | yes |
| 4096 | 8 | 4 | 1 | 7.812500e-03 | 4.768372e-06 | **no** |

RL-Kernel does not remove this dependence (per-KV-group execution would, at 2.7-3.9x
forward time). It binds the degree instead: `AttentionContract.cross_rank_fingerprint`
includes `tp_world_size`, and `validate_cross_config_alignment` fails closed when training
and rollout disagree.

## Reproduce

```bash
python benchmarks/benchmark_rocm_attention.py \
  --output benchmarks/results/pr319_rocm_mi300x/results.json
```
