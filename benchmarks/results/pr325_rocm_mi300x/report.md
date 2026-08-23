# PR #325 ROCm-native Triton distributed FFN correctness and performance

> Operator-only benchmark. No model checkpoint or serving engine was used.

## Environment

| Item | Value |
|---|---|
| NCCL_IB_DISABLE | 1 |
| architecture | gfx942:sramecc+:xnack- |
| deterministic_compute | ROCm-native Triton |
| git_commit | 5222b68f2dc8a45cc07604f67bda31469aaf7aef |
| gpu | AMD Instinct MI300X |
| gpu_count | 8 |
| hip | 7.14.60850 |
| native_collective | ProcessGroupNCCL (RCCL on ROCm) |
| native_gemm | torch.matmul (ROCm rocBLAS/hipBLASLt dispatch) |
| python | 3.12.3 |
| torch | 2.12.0+rocm7.14.0a20260608 |

## Methodology

- BF16 inputs; production FFN dimensions H=4096 and I=12288.
- Native compute is PyTorch `torch.matmul`/elementwise ROCm dispatch; the deterministic compute path is written directly in Triton.
- Native communication is ProcessGroupNCCL (RCCL on ROCm); the deterministic transport is a fixed rank-order RCCL all-gather followed by a balanced BF16 reduction tree.
- Native and Triton distributed FFNs use identical weights, TP/CP/SP sharding, and collective placement.
- Exactness is accepted separately from native numerical proximity: the topology suite compares Triton TP/CP/SP outputs and all gradients bitwise against Triton TP=1.
- Single-GPU timing: GPU events, median and p95; distributed timing: synchronized wall clock, slowest rank/sample.
- 3 warmups, 10 measured forward/collective samples, and 5 forward+backward samples.
- `NCCL_IB_DISABLE=1` forces the same intra-node XGMI transport. Raw median, p95, min, and max values are in `results.json`.

Reproduce from the repository root:

```bash
python benchmarks/benchmark_rocm_ffn.py \
  --warmup 3 \
  --samples 10 \
  --training-samples 5 \
  --output-dir benchmarks/results/pr325_rocm_mi300x
```

Reproduce topology exactness (including Qwen3-8B H=4096/I=12288 TP2):

```bash
NCCL_IB_DISABLE=1 pytest -q \
  tests/distributed/test_qwen_ffn_topology.py
```

## Key findings

- Deterministic Triton GEMM costs 5.6-31.3x native ROCm latency over Qwen3 gate/up and down projection shapes.
- Triton SwiGLU costs 0.73-1.62x native PyTorch latency; the complete single-GPU FFN costs 8.4-24.6x.
- Deterministic collectives cost 2.1-21.0x native RCCL latency across 2/4/8 ranks and 64 KiB/1 MiB/16 MiB per rank.
- Distributed Triton FFN costs 6.5-8.8x native for forward and 4.4-6.5x for forward+backward across TP2/4/8, CP2, and sequence-parallel configurations.
- Deterministic repeats produced 0 mismatched elements; training and inference produced 0 mismatched elements.
- TP2/TP4/TP8, TP+CP, and sequence-parallel topology tests produced 0 output/gradient mismatches versus TP=1; a production-dimension Qwen3-8B TP2 smoke test also produced 0 mismatches.
- Native-vs-Triton error quantifies the accuracy price of fixing every BF16 arithmetic/reduction tree; it is not used as the determinism acceptance criterion.

## Single-GPU GEMM

| Shape | Native ROCm (ms) | Triton (ms) | Triton/native | Repeat mismatch | Triton rel-L2 vs FP32 | Native rel-L2 vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| gate_up_m1 | 0.0387 | 0.2168 | 5.6x | 0 | 5.003e-03 | 1.651e-03 |
| down_m1 | 0.0304 | 0.2824 | 9.3x | 0 | 5.769e-03 | 1.635e-03 |
| gate_up_m8 | 0.0299 | 0.2601 | 8.7x | 0 | 5.018e-03 | 1.663e-03 |
| down_m8 | 0.0325 | 0.3194 | 9.8x | 0 | 5.679e-03 | 1.654e-03 |
| gate_up_m32 | 0.0321 | 0.4611 | 14.4x | 0 | 5.019e-03 | 1.654e-03 |
| down_m32 | 0.0301 | 0.5153 | 17.1x | 0 | 5.604e-03 | 1.662e-03 |
| gate_up_m128 | 0.0481 | 1.2600 | 26.2x | 0 | 5.019e-03 | 1.657e-03 |
| down_m128 | 0.0478 | 1.4965 | 31.3x | 0 | 5.623e-03 | 1.659e-03 |

## Single-GPU SwiGLU

| Case | Native PyTorch (ms) | Triton (ms) | Triton/native | Repeat mismatch | Triton/native rel-L2 |
|---|---:|---:|---:|---:|---:|
| swiglu_forward_m1 | 0.0181 | 0.0294 | 1.62x | 0 | 2.877e-03 |
| swiglu_backward_m1 | 0.0610 | 0.0518 | 0.85x | 0 | 5.159e-03 |
| swiglu_forward_m8 | 0.0213 | 0.0294 | 1.38x | 0 | 2.816e-03 |
| swiglu_backward_m8 | 0.0731 | 0.0536 | 0.73x | 0 | 5.074e-03 |
| swiglu_forward_m32 | 0.0202 | 0.0294 | 1.46x | 0 | 2.825e-03 |
| swiglu_backward_m32 | 0.0715 | 0.0532 | 0.74x | 0 | 5.071e-03 |
| swiglu_forward_m128 | 0.0205 | 0.0287 | 1.40x | 0 | 2.828e-03 |
| swiglu_backward_m128 | 0.0672 | 0.0510 | 0.76x | 0 | 5.059e-03 |

## Single-GPU FFN

| Case | Native ROCm (ms) | Triton (ms) | Triton/native | Repeat mismatch | Train/infer mismatch | Output rel-L2 | Max grad rel-L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ffn_forward_m1 | 0.0973 | 1.8769 | 19.3x | 0 | 0 | 1.008e-02 | nan |
| ffn_train_fwd_bwd_m1 | 0.4162 | 4.4011 | 10.6x | 0 | 0 | 1.008e-02 | 1.037e-02 |
| ffn_forward_m8 | 0.0963 | 2.0095 | 20.9x | 0 | 0 | 9.930e-03 | nan |
| ffn_train_fwd_bwd_m8 | 0.5569 | 4.6699 | 8.4x | 0 | 0 | 9.930e-03 | 1.017e-02 |
| ffn_forward_m32 | 0.1019 | 2.5106 | 24.6x | 0 | 0 | 9.952e-03 | nan |
| ffn_train_fwd_bwd_m32 | 0.4499 | 5.7878 | 12.9x | 0 | 0 | 9.952e-03 | 1.024e-02 |

## RCCL collectives

| Operation | Ranks | Input/rank | Native RCCL (ms) | Deterministic (ms) | Overhead | Repeat mismatch |
|---|---:|---:|---:|---:|---:|---:|
| all_reduce | 2 | 0.0625 MiB | 0.0759 | 0.5835 | 7.7x | 0 |
| all_gather | 2 | 0.0625 MiB | 0.0516 | 0.5461 | 10.6x | 0 |
| reduce_scatter | 2 | 0.0625 MiB | 0.0545 | 0.5850 | 10.7x | 0 |
| all_reduce | 2 | 1 MiB | 0.0901 | 0.5708 | 6.3x | 0 |
| all_gather | 2 | 1 MiB | 0.0721 | 0.5574 | 7.7x | 0 |
| reduce_scatter | 2 | 1 MiB | 0.0734 | 0.5869 | 8.0x | 0 |
| all_reduce | 2 | 16 MiB | 0.4251 | 0.8872 | 2.1x | 0 |
| all_gather | 2 | 16 MiB | 0.4061 | 0.8974 | 2.2x | 0 |
| reduce_scatter | 2 | 16 MiB | 0.2442 | 0.8994 | 3.7x | 0 |
| all_reduce | 4 | 0.0625 MiB | 0.0767 | 0.7018 | 9.1x | 0 |
| all_gather | 4 | 0.0625 MiB | 0.0588 | 0.6768 | 11.5x | 0 |
| reduce_scatter | 4 | 0.0625 MiB | 0.0576 | 0.7142 | 12.4x | 0 |
| all_reduce | 4 | 1 MiB | 0.0908 | 0.7026 | 7.7x | 0 |
| all_gather | 4 | 1 MiB | 0.0887 | 0.6913 | 7.8x | 0 |
| reduce_scatter | 4 | 1 MiB | 0.0886 | 0.7357 | 8.3x | 0 |
| all_reduce | 4 | 16 MiB | 0.2568 | 1.0472 | 4.1x | 0 |
| all_gather | 4 | 16 MiB | 0.4271 | 1.0324 | 2.4x | 0 |
| reduce_scatter | 4 | 16 MiB | 0.1546 | 1.0564 | 6.8x | 0 |
| all_reduce | 8 | 0.0625 MiB | 0.0490 | 1.0286 | 21.0x | 0 |
| all_gather | 8 | 0.0625 MiB | 0.0718 | 0.9602 | 13.4x | 0 |
| reduce_scatter | 8 | 0.0625 MiB | 0.0705 | 1.0240 | 14.5x | 0 |
| all_reduce | 8 | 1 MiB | 0.0726 | 1.0293 | 14.2x | 0 |
| all_gather | 8 | 1 MiB | 0.1039 | 1.0011 | 9.6x | 0 |
| reduce_scatter | 8 | 1 MiB | 0.0916 | 1.0526 | 11.5x | 0 |
| all_reduce | 8 | 16 MiB | 0.1957 | 1.3791 | 7.0x | 0 |
| all_gather | 8 | 16 MiB | 0.4314 | 1.3418 | 3.1x | 0 |
| reduce_scatter | 8 | 16 MiB | 0.1210 | 1.3716 | 11.3x | 0 |

## Distributed FFN

| Topology | Direction | Native ROCm/RCCL (ms) | Triton/deterministic RCCL (ms) | Overhead | Repeat mismatch | Train/infer mismatch | Output rel-L2 | Max grad rel-L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tp2 | forward | 0.2215 | 1.9557 | 8.8x | 0 | 0 | 1.012e-02 | nan |
| tp2 | train_fwd_bwd | 1.1951 | 6.1529 | 5.1x | 0 | 0 | 1.012e-02 | 1.079e-02 |
| tp2_sp | forward | 0.3040 | 2.5241 | 8.3x | 0 | 0 | 1.015e-02 | nan |
| tp2_sp | train_fwd_bwd | 0.9795 | 6.3449 | 6.5x | 0 | 0 | 1.015e-02 | 1.083e-02 |
| tp4 | forward | 0.2191 | 1.4863 | 6.8x | 0 | 0 | 1.031e-02 | nan |
| tp4 | train_fwd_bwd | 0.8463 | 4.4050 | 5.2x | 0 | 0 | 1.031e-02 | 1.095e-02 |
| tp2_cp2 | forward | 0.2288 | 1.8150 | 7.9x | 0 | 0 | 1.015e-02 | nan |
| tp2_cp2 | train_fwd_bwd | 1.7761 | 8.7733 | 4.9x | 0 | 0 | 1.015e-02 | 1.083e-02 |
| tp2_cp2_sp | forward | 0.2944 | 2.3455 | 8.0x | 0 | 0 | 1.022e-02 | nan |
| tp2_cp2_sp | train_fwd_bwd | 1.9599 | 8.7198 | 4.4x | 0 | 0 | 1.022e-02 | 1.089e-02 |
| tp8 | forward | 0.2065 | 1.7302 | 8.4x | 0 | 0 | 1.060e-02 | nan |
| tp8 | train_fwd_bwd | 1.1059 | 7.1109 | 6.4x | 0 | 0 | 1.060e-02 | 1.126e-02 |
| tp4_cp2 | forward | 0.2255 | 1.4714 | 6.5x | 0 | 0 | 1.033e-02 | nan |
| tp4_cp2 | train_fwd_bwd | 1.3987 | 8.5660 | 6.1x | 0 | 0 | 1.033e-02 | 1.099e-02 |
| tp4_cp2_sp | forward | 0.2878 | 2.1547 | 7.5x | 0 | 0 | 1.045e-02 | nan |
| tp4_cp2_sp | train_fwd_bwd | 1.8003 | 8.8284 | 4.9x | 0 | 0 | 1.045e-02 | 1.109e-02 |

## Communication overlap feasibility

The measured implementation deliberately serializes dependencies; the numbers above do not claim overlap. A representative 1 MiB deterministic collective costs 0.5574-1.0526 ms, while distributed Triton FFN forward costs 1.4714-2.5241 ms and forward+backward costs 4.4050-8.8284 ms.

- Forward SP all-gather feeds both gate/up GEMMs, and the final TP all-reduce or reduce-scatter consumes the down projection. These are hard data dependencies.
- Backward can overlap the fixed-order TP reduction of the gate contribution to dHidden with computation of the independent up contribution. The final wait and gate-then-up add order must remain unchanged to preserve 0 mismatch.
- CP gathers are prerequisites for weight-gradient GEMMs. Packing them in a fixed layout can amortize launch/signature overhead, but they cannot be hidden behind the GEMMs that consume them.
- Coalescing and host-side validation caching should be measured before introducing multi-stream scheduling complexity.

## Figures

![Single-GPU overhead](single_gpu_overhead.png)

![RCCL collective overhead](collective_overhead.png)

![Distributed FFN overhead](distributed_ffn_overhead.png)

![Accuracy trade-off](accuracy_tradeoff.png)
