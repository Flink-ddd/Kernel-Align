# PR #325 ROCm deterministic Triton FFN report

This is an operator-only MI300X report. It does not load or benchmark a model checkpoint.

## Comparison contract

1. **Determinism:** every Triton TP/CP/SP result is compared bitwise with the same deterministic Triton FFN at **TP=1**. The reported metric is element mismatch count; acceptance requires 0.
2. **FP16/FP32:** one separate, simple output comparison runs official Hugging Face `Qwen3MLP` at TP=1 in FP16 and FP32. FP32 is the reference.
3. **Speed:** official Hugging Face `Qwen3MLP` at TP=1 is the only speed baseline. Speed tables and figures intentionally contain no numerical accuracy comparison between official FFN and Triton.

## Environment

| Field | Value |
|---|---|
| NCCL_IB_DISABLE | 1 |
| architecture | gfx942:sramecc+:xnack- |
| deterministic_compute | ROCm-native Triton |
| deterministic_transport | fixed-order RCCL on ROCm |
| git_commit | 08f47d97d0443c5998b8da6b41a22fdf3848da8f |
| gpu | AMD Instinct MI300X |
| gpu_count | 8 |
| hip | 7.14.60850 |
| python | 3.12.3 |
| speed_baseline | Hugging Face Transformers Qwen3MLP, TP=1 |
| torch | 2.12.0+rocm7.14.0a20260608 |
| transformers | 5.10.4 |

## Methodology

- Operator shape: H=4096, I=12288; BF16 is used for all speed and determinism measurements.
- Single-GPU shapes use M=1/8/32. Distributed cases use the same full logical M=32 input for TP2/4/8, TP+CP, and sequence parallelism.
- The official performance baseline is upstream Transformers `Qwen3MLP` with unsharded weights and input (TP=1). Each distributed rank runs that TP=1 reference and the slowest rank/sample is reported.
- The distributed exactness baseline is the PR's deterministic Triton FFN at TP=1. Local outputs, dHidden, and sharded dWeights are compared against their exact TP=1 slices.
- Single-GPU timing: GPU events, median and p95; distributed timing: synchronized wall clock, slowest rank/sample.
- 3 warmups, 10 measured forward samples, and 5 measured forward+backward samples.
- `NCCL_IB_DISABLE=1` keeps the distributed run on intra-node XGMI. Median, p95, min, and max values are available in `results.json`.

Reproduce from the repository root:

```bash
python benchmarks/benchmark_rocm_ffn.py \
  --warmup 3 \
  --samples 10 \
  --training-samples 5 \
  --output-dir benchmarks/results/pr325_rocm_mi300x
```

## Results summary

- TP=1 exactness baseline: **0 mismatched elements** across topology forward outputs, training outputs, dHidden, and dWeights.
- Repeat mismatch: **0**; training/inference forward mismatch: **0**.
- Single-GPU deterministic Triton latency is **7.38-22.86x** the official Qwen3MLP TP=1 latency across M=1/8/32 and forward/training.
- Distributed deterministic Triton latency is **7.45-15.79x** the official Qwen3MLP TP=1 latency across tested parallel layouts.
- The separate official-Qwen3MLP FP16 versus FP32 observation has relative-L2 error **6.544e-04** for (M,H,I)=(8,4096,12288).

## Single-GPU FFN speed

Performance only; no official-versus-Triton accuracy metric is reported here.

| Shape / direction | Official Qwen3MLP TP=1 (ms) | Deterministic Triton (ms) | Triton / official TP=1 |
|---|---:|---:|---:|
| (M,H,I)=(1,4096,12288), forward | 0.1183 | 1.9102 | 16.15x |
| (M,H,I)=(1,4096,12288), forward+backward | 0.6008 | 4.4312 | 7.38x |
| (M,H,I)=(8,4096,12288), forward | 0.2263 | 2.0424 | 9.03x |
| (M,H,I)=(8,4096,12288), forward+backward | 0.3997 | 4.6210 | 11.56x |
| (M,H,I)=(32,4096,12288), forward | 0.1108 | 2.5334 | 22.86x |
| (M,H,I)=(32,4096,12288), forward+backward | 0.7081 | 5.7710 | 8.15x |

## Distributed FFN speed

The baseline remains the full logical M=32 official FFN at TP=1 for every row. Performance only; no numerical accuracy is mixed into this comparison.

| Parallel layout | Direction | Official Qwen3MLP TP=1 (ms) | Deterministic distributed Triton (ms) | Triton / official TP=1 |
|---|---|---:|---:|---:|
| tp2 | forward | 0.1575 | 1.9598 | 12.44x |
| tp2 | train_fwd_bwd | 0.6511 | 5.1613 | 7.93x |
| tp2_sp | forward | 0.1575 | 2.4866 | 15.79x |
| tp2_sp | train_fwd_bwd | 0.6511 | 9.0512 | 13.90x |
| tp4 | forward | 0.1644 | 1.5044 | 9.15x |
| tp4 | train_fwd_bwd | 0.7244 | 5.3938 | 7.45x |
| tp2_cp2 | forward | 0.1644 | 1.8202 | 11.07x |
| tp2_cp2 | train_fwd_bwd | 0.7244 | 10.1842 | 14.06x |
| tp2_cp2_sp | forward | 0.1644 | 2.3788 | 14.47x |
| tp2_cp2_sp | train_fwd_bwd | 0.7244 | 9.7862 | 13.51x |
| tp8 | forward | 0.1692 | 1.7434 | 10.31x |
| tp8 | train_fwd_bwd | 0.6325 | 5.1244 | 8.10x |
| tp4_cp2 | forward | 0.1692 | 1.4826 | 8.76x |
| tp4_cp2 | train_fwd_bwd | 0.6325 | 7.2102 | 11.40x |
| tp4_cp2_sp | forward | 0.1692 | 2.1856 | 12.92x |
| tp4_cp2_sp | train_fwd_bwd | 0.6325 | 8.6635 | 13.70x |

## CUDA GPU and CPU performance context

The additional measurements come from [PR #321 deterministic CUDA FFN performance report](https://github.com/RL-Align/RL-Kernel/pull/321) at CUDA commit `8576fa4bf449734ae99e9b50be8756bb282a8916`. H100 Triton replays use this PR's code at `e64abab904880b877d26d04c0cfad020b992aa51`.

The same-H100 CUDA/Triton ratio is the hardware-matched comparison. CPU and MI300X columns provide absolute-latency context only; they are not hardware-normalized speed claims.

| Comparison environment | Value |
|---|---|
| CUDA GPU | NVIDIA H100 80GB HBM3 (sm_90) |
| CUDA / PyTorch | 13.0 / 2.13.0+cu130 |
| CPU | Intel(R) Xeon(R) Platinum 8468, 96 intra-op threads |
| Transformers | 5.13.1 |

### Single-GPU and CPU absolute latency

| Shape / direction | CPU official (ms) | H100 official TP=1 (ms) | H100 Triton replay (ms) | H100 CUDA (ms) | CUDA / Triton H100 | MI300X official TP=1 (ms) | MI300X Triton (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| M=1, forward | 12.9558 | 0.1193 | 1.7381 | 3.9988 | 2.30x | 0.1183 | 1.9102 |
| M=1, forward+backward | 58.5765 | 0.5184 | 4.2997 | 9.0704 | 2.11x | 0.6008 | 4.4312 |
| M=8, forward | 12.7676 | 0.1239 | 1.8293 | 3.9923 | 2.18x | 0.2263 | 2.0424 |
| M=8, forward+backward | 82.3669 | 0.5414 | 4.5028 | 9.4500 | 2.10x | 0.3997 | 4.6210 |
| M=32, forward | 8.5392 | 0.1311 | 2.2529 | 4.0277 | 1.79x | 0.1108 | 2.5334 |
| M=32, forward+backward | 65.9689 | 0.7842 | 5.4261 | 9.1635 | 1.69x | 0.7081 | 5.7710 |

### Distributed absolute latency

No distributed CPU or H100 Triton replay measurement was supplied. Each deterministic implementation is therefore compared with its own hardware's official TP=1 baseline.

| Layout | Direction | H100 official TP=1 (ms) | H100 CUDA (ms) | CUDA / H100 official | MI300X official TP=1 (ms) | MI300X Triton (ms) | Triton / MI300X official |
|---|---|---:|---:|---:|---:|---:|---:|
| tp2 | forward | 0.1552 | 2.7896 | 17.97x | 0.1575 | 1.9598 | 12.44x |
| tp2 | train_fwd_bwd | 0.6301 | 7.8125 | 12.40x | 0.6511 | 5.1613 | 7.93x |
| tp2_sp | forward | 0.1552 | 3.3310 | 21.46x | 0.1575 | 2.4866 | 15.79x |
| tp2_sp | train_fwd_bwd | 0.6301 | 11.7695 | 18.68x | 0.6511 | 9.0512 | 13.90x |
| tp4 | forward | 0.1530 | 2.1687 | 14.17x | 0.1644 | 1.5044 | 9.15x |
| tp4 | train_fwd_bwd | 0.5714 | 9.3954 | 16.44x | 0.7244 | 5.3938 | 7.45x |
| tp2_cp2 | forward | 0.1530 | 2.8261 | 18.47x | 0.1644 | 1.8202 | 11.07x |
| tp2_cp2 | train_fwd_bwd | 0.5714 | 16.1722 | 28.30x | 0.7244 | 10.1842 | 14.06x |
| tp2_cp2_sp | forward | 0.1530 | 3.5814 | 23.41x | 0.1644 | 2.3788 | 14.47x |
| tp2_cp2_sp | train_fwd_bwd | 0.5714 | 16.7894 | 29.38x | 0.7244 | 9.7862 | 13.51x |
| tp8 | forward | 0.1537 | 2.3206 | 15.10x | 0.1692 | 1.7434 | 10.31x |
| tp8 | train_fwd_bwd | 0.5006 | 10.1375 | 20.25x | 0.6325 | 5.1244 | 8.10x |
| tp4_cp2 | forward | 0.1537 | 2.2141 | 14.41x | 0.1692 | 1.4826 | 8.76x |
| tp4_cp2 | train_fwd_bwd | 0.5006 | 16.5617 | 33.08x | 0.6325 | 7.2102 | 11.40x |
| tp4_cp2_sp | forward | 0.1537 | 3.2972 | 21.45x | 0.1692 | 2.1856 | 12.92x |
| tp4_cp2_sp | train_fwd_bwd | 0.5006 | 17.1457 | 34.25x | 0.6325 | 8.6635 | 13.70x |

## Topology exactness versus Triton TP=1

All columns are element mismatch counts. This table does not compare against the official FFN.

| Parallel layout | Forward output | Training output | dHidden | dWeights | Repeat | Train/infer |
|---|---:|---:|---:|---:|---:|---:|
| tp2 | 0 | 0 | 0 | 0 | 0 | 0 |
| tp2_sp | 0 | 0 | 0 | 0 | 0 | 0 |
| tp4 | 0 | 0 | 0 | 0 | 0 | 0 |
| tp2_cp2 | 0 | 0 | 0 | 0 | 0 | 0 |
| tp2_cp2_sp | 0 | 0 | 0 | 0 | 0 | 0 |
| tp8 | 0 | 0 | 0 | 0 | 0 | 0 |
| tp4_cp2 | 0 | 0 | 0 | 0 | 0 | 0 |
| tp4_cp2_sp | 0 | 0 | 0 | 0 | 0 | 0 |

## Simple FP16 versus FP32 observation

This is an official `Qwen3MLP` TP=1 output comparison only; it is not used to judge deterministic Triton and is not included in speed ratios.

| Shape | Candidate | Reference | Max abs | Mean abs | Relative L2 |
|---|---|---|---:|---:|---:|
| (M,H,I)=(8,4096,12288) | FP16 | FP32 | 2.046e-06 | 3.742e-07 | 6.544e-04 |

## Deterministic communication overlap

The current timing includes the fixed-order communication schedule and makes no overlap claim. Forward SP all-gather must finish before gate/up projection, and TP reduction consumes the down-projection output, so those edges are hard dependencies.

In backward, the gate and up contributions to dHidden are independent until their final ordered addition. A future implementation can place the fixed-rank reduction of one contribution on a second stream while computing the other, but it must preserve rank order, reduction tree, wait points, and gate-then-up addition order. Any optimization is accepted only if every TP=1 mismatch column remains zero.

## Figures

![Single-GPU CUDA, Triton, and CPU latency](single_gpu_overhead.png)

![Topology mismatch versus Triton TP=1](collective_overhead.png)

![Distributed H100 CUDA and MI300X Triton latency](distributed_ffn_overhead.png)
