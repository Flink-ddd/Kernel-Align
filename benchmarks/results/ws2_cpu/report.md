# WS2 strict ROCm Attention — bitwise parity and performance

> Operator-only benchmark. No model checkpoint or serving engine was used;
> the shapes are Qwen3-8B's attention shapes.

## Environment

| Item | cpu |
|---|---|
| architecture | n/a |
| cpu_count | 192 |
| cuda | None |
| extension_attention_symbols | deterministic_attention_backward, deterministic_attention_forward |
| gpu | n/a (host execution) |
| gpu_count | 8 |
| hip | 7.14.60850 |
| native_collective | torch.distributed ProcessGroupNCCL (RCCL on ROCm) |
| python | 3.12.3 |
| torch | 2.12.0+rocm7.14.0a20260608 |
| torch_threads | 32 |
| triton | 3.7.0 |

## Methodology

- Operator shape: `Hq=32`, `Hkv=8`, `D=128`, `B=1`, causal; sequence sweep 512, 1024, 2048.
- Measured paths:
  - `sdpa`: `torch.nn.functional.scaled_dot_product_attention`. **Speed baseline only** — as in PR #325, no accuracy comparison is mixed into the speed table.
  - `strict-aiter`: `StrictRocmAiterCKAttentionCore` called **once for all heads**. This is the core, not the production schedule: the Vime provider launches it once per (batch row, KV group). See the per-KV-group schedule table for that cost.
  - `reference-native`: `_C.deterministic_attention_forward/backward`, the materializing FP32 reference core hipified from the shared `.cu`.
  - `triton-bitwise`: `TritonDeterministicAttentionOp`, whose contract is bit-identity with `reference-native`.
- Timing: CUDA events, median and p95. Peak memory is the per-call increase in `torch.cuda.max_memory_allocated` above what was live before the call.
- Accuracy is against an FP64 oracle over the same BF16/FP16-rounded inputs. Repeat = two identical calls are bitwise equal; batch-invariant = a row computed alone is bitwise equal to the same row inside a batch.
- 2 warmups, 5 measured forward samples, 3 measured forward+backward samples. Raw medians, p95, min and max are in `results.json`.

Reproduce from the repository root:

```bash
python benchmarks/benchmark_ws2_rocm_attention.py \
  --seq-lens 512,1024,2048 \
  --dtypes bf16,fp16 \
  --warmup 2 --samples 5 \
  --training-samples 3 \
  --output-dir benchmarks/results/ws2_rocm_mi300x
```

### Unavailable paths

- `strict-aiter`: GPU-only path; not available on the host
- `strict-fa4`: GPU-only path; not available on the host
- `reference-native`: GPU-only path; not available on the host
- `triton-bitwise`: GPU-only path; not available on the host

## Bitwise parity: Triton port vs the native reference core

Acceptance is 0 mismatched elements. This is the contract the Triton core exists to hold.

| dtype | S | out mismatched | lse mismatched | dQ | dK | dV | bitwise |
|---|---:|---:|---:|---:|---:|---:|:---:|

`dQ/dK/dV` are measured on the BF16 sweep only; `n/a` marks the FP16 rows.

## Single-device Attention (bf16)

### Forward

| S | Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB | out max-abs vs FP64 | lse max-abs vs FP64 | Repeat |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 512 | sdpa | 31.5270 | 34.1950 | 1.00x | 0.0 | 8.606e-03 | n/a | yes |
| 512 | pytorch-native | 15.3921 | 15.8488 | 0.49x | 32.7 | 1.947e-02 | n/a | yes |
| 1024 | sdpa | 98.3129 | 98.4924 | 1.00x | 43.6 | 8.191e-03 | n/a | yes |
| 1024 | pytorch-native | 51.5739 | 51.9609 | 0.52x | 127.8 | 1.610e-02 | n/a | yes |
| 2048 | sdpa | 304.6115 | 308.9697 | 1.00x | 50.5 | 9.314e-03 | n/a | yes |
| 2048 | pytorch-native | 216.1569 | 235.2967 | 0.71x | 543.6 | 1.790e-02 | n/a | yes |

### Forward+backward

| S | Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB |
|---:|---|---:|---:|---:|---:|
| 512 | sdpa | 53.7609 | 54.4313 | 1.00x | 0.0 |
| 512 | pytorch-native | 24.7810 | 25.4956 | 0.46x | 46.4 |
| 1024 | sdpa | 163.8662 | 167.7328 | 1.00x | 30.9 |
| 1024 | pytorch-native | 107.9884 | 110.4447 | 0.66x | 191.6 |
| 2048 | sdpa | 533.5067 | 648.9086 | 1.00x | 31.7 |
| 2048 | pytorch-native | 422.9041 | 424.9965 | 0.79x | 776.1 |

## Single-device Attention (fp16)

### Forward

| S | Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB | out max-abs vs FP64 | lse max-abs vs FP64 | Repeat |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 512 | sdpa | 35.0730 | 38.8791 | 1.00x | 0.0 | 1.045e-03 | n/a | yes |
| 512 | pytorch-native | 734.5447 | 1132.5722 | 20.94x | 0.0 | 2.811e-03 | n/a | yes |
| 1024 | sdpa | 105.0845 | 109.9980 | 1.00x | 0.0 | 1.075e-03 | n/a | yes |
| 1024 | pytorch-native | 3093.5610 | 3837.3499 | 29.44x | 128.0 | 2.117e-03 | n/a | yes |
| 2048 | sdpa | 1566.1332 | 1683.1755 | 1.00x | 51.0 | 1.065e-03 | n/a | yes |
| 2048 | pytorch-native | 13198.5082 | 13483.8688 | 8.43x | 561.1 | 2.143e-03 | n/a | yes |

### Forward+backward

| S | Path | Median (ms) | p95 (ms) | vs sdpa | Peak MiB |
|---:|---|---:|---:|---:|---:|
| 512 | sdpa | 156.9364 | 240.0815 | 1.00x | 0.0 |
| 512 | pytorch-native | 2364.2130 | 3938.9163 | 15.06x | 0.0 |
| 1024 | sdpa | 335.7934 | 337.1536 | 1.00x | 79.1 |
| 1024 | pytorch-native | 11104.1810 | 13321.1288 | 33.07x | 192.2 |
| 2048 | sdpa | 1089.9534 | 1090.4449 | 1.00x | 143.9 |
| 2048 | pytorch-native | 37002.9627 | 38977.8489 | 33.95x | 766.4 |

## Production core versus the reference core

These are two different kernels, so this is a tolerance comparison, not a parity claim. It is here to size the gap, not to assert equality.

| dtype | S | out max-abs | out relative-L2 | lse max-abs |
|---|---:|---:|---:|---:|

## Batch-composition invariance

A row computed alone must be bitwise equal to the same row inside a batch. The strict ROCm core rejects `B > 1` outright, so for that path the property is structural rather than measured.

| S | Path | Bitwise | Mismatched | Note |
|---:|---|:---:|---:|---|
| 512 | sdpa | yes | 0 | measured |
| 512 | pytorch-native | yes | 0 | measured |
| 1024 | sdpa | yes | 0 | measured |
| 1024 | pytorch-native | yes | 0 | measured |
| 2048 | sdpa | yes | 0 | measured |
| 2048 | pytorch-native | yes | 0 | measured |

## TP-degree invariance of the strict ROCm core

A head shard computed under TP=N versus the same slice of an unsharded run. TP performs no cross-rank reduction in attention, so any nonzero value means the kernel's result depends on how many heads shared the launch. `raw_launch` is one launch for all heads; `one_kv_group_per_launch` is the schedule the Vime provider actually uses.

| S | Schedule | TP | Local Hq | Local Hkv | out max-abs | lse max-abs | Invariant |
|---:|---|---:|---:|---:|---:|---:|:---:|

## Figures

`reference-native` and `triton-bitwise` allocate exactly the same buffers, so their memory curves coincide and the later-drawn series hides the earlier one.

![Single-device latency and memory grid](single_gpu_grid.png)

![Single-device latency](single_gpu_latency.png)

![Single-device peak memory](single_gpu_memory.png)

![Bitwise exactness matrix](exactness_matrix.png)

![TP-degree invariance](tp_degree_invariance.png)

