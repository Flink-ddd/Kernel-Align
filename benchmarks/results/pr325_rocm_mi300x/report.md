# PR #325 ROCm-native Triton distributed FFN correctness and performance

> Operator-only benchmark. No model checkpoint or serving engine was used.

## Environment

| Item | Value |
|---|---|
| gpu | AMD Instinct MI300X |
| gpu_count | 8 |
| architecture | gfx942:sramecc+:xnack- |
| torch | 2.12.0+rocm7.14.0a20260608 |
| hip | 7.14.60850 |
| python | 3.12.3 |
| git_commit | 43a31ae7f6535e2c11efdd50d96b6633805359fd |
| native_gemm | torch.matmul (ROCm rocBLAS/hipBLASLt dispatch) |
| deterministic_compute | ROCm-native Triton |
| native_collective | ProcessGroupNCCL (RCCL on ROCm) |
| NCCL_IB_DISABLE | 1 |

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

- Deterministic Triton GEMM costs 5.4-31.5x native ROCm latency over the tested `(M,K,N)` shapes.
- Triton SwiGLU costs 0.72-1.60x native PyTorch latency; the complete single-GPU FFN costs 7.4-23.1x.
- Deterministic collectives cost 2.1-19.9x native RCCL latency across 2/4/8 ranks and 64 KiB/1 MiB/16 MiB per rank.
- Distributed Triton FFN costs 6.5-9.1x native for forward and 4.4-6.8x for forward+backward across TP2/4/8, CP2, and sequence-parallel configurations.
- Deterministic repeats produced 0 mismatched elements; training and inference produced 0 mismatched elements.
- TP2/TP4/TP8, TP+CP, and sequence-parallel topology tests produced 0 output/gradient mismatches versus TP=1; a production-dimension Qwen3-8B TP2 smoke test also produced 0 mismatches.
- Numerical drift means Triton versus native ROCm/RCCL on identical BF16 inputs. GEMM-to-FP32 error is separately labelled and uses `A.float() @ B.float()` as its reference.

## Accuracy references

- `Repeat mismatch`: Triton run 1 versus Triton run 2.
- `Train/infer mismatch`: Triton training forward versus Triton inference forward.
- `Topology mismatch`: distributed Triton TP/CP/SP versus Triton TP=1.
- `vs FP32`: BF16 GEMM output versus `A.float() @ B.float()`.
- `Triton/native rel-L2`: Triton output or gradient versus the native ROCm/RCCL path with identical BF16 tensors and parallel layout.

## Single-GPU GEMM

| Shape | Native ROCm (ms) | Triton (ms) | Triton/native | Repeat mismatch | Triton rel-L2 vs FP32 | Native rel-L2 vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| (M,K,N)=(1,4096,12288) | 0.0388 | 0.2087 | 5.4x | 0 | 5.003e-03 | 1.651e-03 |
| (M,K,N)=(1,12288,4096) | 0.0299 | 0.2629 | 8.8x | 0 | 5.769e-03 | 1.635e-03 |
| (M,K,N)=(8,4096,12288) | 0.0305 | 0.2543 | 8.3x | 0 | 5.018e-03 | 1.663e-03 |
| (M,K,N)=(8,12288,4096) | 0.0315 | 0.3225 | 10.2x | 0 | 5.679e-03 | 1.654e-03 |
| (M,K,N)=(32,4096,12288) | 0.0321 | 0.4758 | 14.8x | 0 | 5.019e-03 | 1.654e-03 |
| (M,K,N)=(32,12288,4096) | 0.0308 | 0.4955 | 16.1x | 0 | 5.604e-03 | 1.662e-03 |
| (M,K,N)=(128,4096,12288) | 0.0472 | 1.2570 | 26.7x | 0 | 5.019e-03 | 1.657e-03 |
| (M,K,N)=(128,12288,4096) | 0.0479 | 1.5075 | 31.5x | 0 | 5.623e-03 | 1.659e-03 |

## Single-GPU SwiGLU

| Case | Native PyTorch (ms) | Triton (ms) | Triton/native | Repeat mismatch | Triton/native ROCm rel-L2 |
|---|---:|---:|---:|---:|---:|
| (M,I)=(1,12288), forward | 0.0176 | 0.0282 | 1.60x | 0 | 2.877e-03 |
| (M,I)=(1,12288), backward | 0.0590 | 0.0507 | 0.86x | 0 | 5.159e-03 |
| (M,I)=(8,12288), forward | 0.0203 | 0.0285 | 1.40x | 0 | 2.816e-03 |
| (M,I)=(8,12288), backward | 0.0684 | 0.0512 | 0.75x | 0 | 5.074e-03 |
| (M,I)=(32,12288), forward | 0.0203 | 0.0280 | 1.38x | 0 | 2.825e-03 |
| (M,I)=(32,12288), backward | 0.0684 | 0.0495 | 0.72x | 0 | 5.071e-03 |
| (M,I)=(128,12288), forward | 0.0196 | 0.0268 | 1.37x | 0 | 2.828e-03 |
| (M,I)=(128,12288), backward | 0.0643 | 0.0493 | 0.77x | 0 | 5.059e-03 |

## Single-GPU FFN

| Case | Native ROCm (ms) | Triton (ms) | Triton/native | Repeat mismatch | Train/infer mismatch | Triton/native output rel-L2 | Triton/native max grad rel-L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| (M,H,I)=(1,4096,12288), forward | 0.1052 | 1.8770 | 17.8x | 0 | 0 | 1.008e-02 | nan |
| (M,H,I)=(1,4096,12288), forward+backward | 0.3984 | 4.4071 | 11.1x | 0 | 0 | 1.008e-02 | 1.037e-02 |
| (M,H,I)=(8,4096,12288), forward | 0.1081 | 2.0033 | 18.5x | 0 | 0 | 9.930e-03 | nan |
| (M,H,I)=(8,4096,12288), forward+backward | 0.6354 | 4.6865 | 7.4x | 0 | 0 | 9.930e-03 | 1.017e-02 |
| (M,H,I)=(32,4096,12288), forward | 0.1085 | 2.5084 | 23.1x | 0 | 0 | 9.952e-03 | nan |
| (M,H,I)=(32,4096,12288), forward+backward | 0.4620 | 5.7915 | 12.5x | 0 | 0 | 9.952e-03 | 1.024e-02 |

## RCCL collectives

| Operation | Ranks | Input/rank | Native RCCL (ms) | Deterministic (ms) | Overhead | Repeat mismatch |
|---|---:|---:|---:|---:|---:|---:|
| all_reduce | 2 | 0.0625 MiB | 0.0746 | 0.5678 | 7.6x | 0 |
| all_gather | 2 | 0.0625 MiB | 0.0525 | 0.5303 | 10.1x | 0 |
| reduce_scatter | 2 | 0.0625 MiB | 0.0514 | 0.5667 | 11.0x | 0 |
| all_reduce | 2 | 1 MiB | 0.0893 | 0.5569 | 6.2x | 0 |
| all_gather | 2 | 1 MiB | 0.0734 | 0.5396 | 7.4x | 0 |
| reduce_scatter | 2 | 1 MiB | 0.0694 | 0.5679 | 8.2x | 0 |
| all_reduce | 2 | 16 MiB | 0.4230 | 0.8854 | 2.1x | 0 |
| all_gather | 2 | 16 MiB | 0.4086 | 0.8916 | 2.2x | 0 |
| reduce_scatter | 2 | 16 MiB | 0.2422 | 0.8880 | 3.7x | 0 |
| all_reduce | 4 | 0.0625 MiB | 0.0794 | 0.6986 | 8.8x | 0 |
| all_gather | 4 | 0.0625 MiB | 0.0584 | 0.6524 | 11.2x | 0 |
| reduce_scatter | 4 | 0.0625 MiB | 0.0563 | 0.7090 | 12.6x | 0 |
| all_reduce | 4 | 1 MiB | 0.0935 | 0.6918 | 7.4x | 0 |
| all_gather | 4 | 1 MiB | 0.0881 | 0.6801 | 7.7x | 0 |
| reduce_scatter | 4 | 1 MiB | 0.0906 | 0.7181 | 7.9x | 0 |
| all_reduce | 4 | 16 MiB | 0.2569 | 1.0410 | 4.1x | 0 |
| all_gather | 4 | 16 MiB | 0.4229 | 1.0316 | 2.4x | 0 |
| reduce_scatter | 4 | 16 MiB | 0.1543 | 1.0400 | 6.7x | 0 |
| all_reduce | 8 | 0.0625 MiB | 0.0517 | 1.0270 | 19.9x | 0 |
| all_gather | 8 | 0.0625 MiB | 0.0745 | 0.9675 | 13.0x | 0 |
| reduce_scatter | 8 | 0.0625 MiB | 0.0675 | 1.0467 | 15.5x | 0 |
| all_reduce | 8 | 1 MiB | 0.0759 | 1.0367 | 13.7x | 0 |
| all_gather | 8 | 1 MiB | 0.1045 | 1.0074 | 9.6x | 0 |
| reduce_scatter | 8 | 1 MiB | 0.0928 | 1.0538 | 11.4x | 0 |
| all_reduce | 8 | 16 MiB | 0.1947 | 1.3829 | 7.1x | 0 |
| all_gather | 8 | 16 MiB | 0.4276 | 1.4123 | 3.3x | 0 |
| reduce_scatter | 8 | 16 MiB | 0.1201 | 1.3751 | 11.5x | 0 |

## Distributed FFN

| Topology | Direction | Native ROCm/RCCL (ms) | Triton/deterministic RCCL (ms) | Overhead | Repeat mismatch | Train/infer mismatch | Triton/native output rel-L2 | Triton/native max grad rel-L2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tp2 | forward | 0.2178 | 1.9718 | 9.1x | 0 | 0 | 1.012e-02 | nan |
| tp2 | train_fwd_bwd | 0.8281 | 5.1330 | 6.2x | 0 | 0 | 1.012e-02 | 1.079e-02 |
| tp2_sp | forward | 0.2785 | 2.5044 | 9.0x | 0 | 0 | 1.015e-02 | nan |
| tp2_sp | train_fwd_bwd | 0.9735 | 6.3391 | 6.5x | 0 | 0 | 1.015e-02 | 1.083e-02 |
| tp4 | forward | 0.2179 | 1.4986 | 6.9x | 0 | 0 | 1.031e-02 | nan |
| tp4 | train_fwd_bwd | 1.0952 | 6.2879 | 5.7x | 0 | 0 | 1.031e-02 | 1.095e-02 |
| tp2_cp2 | forward | 0.2206 | 1.8216 | 8.3x | 0 | 0 | 1.015e-02 | nan |
| tp2_cp2 | train_fwd_bwd | 1.1910 | 7.5867 | 6.4x | 0 | 0 | 1.015e-02 | 1.083e-02 |
| tp2_cp2_sp | forward | 0.2823 | 2.3629 | 8.4x | 0 | 0 | 1.022e-02 | nan |
| tp2_cp2_sp | train_fwd_bwd | 1.2807 | 8.7284 | 6.8x | 0 | 0 | 1.022e-02 | 1.089e-02 |
| tp8 | forward | 0.1965 | 1.7419 | 8.9x | 0 | 0 | 1.060e-02 | nan |
| tp8 | train_fwd_bwd | 1.0607 | 7.2079 | 6.8x | 0 | 0 | 1.060e-02 | 1.126e-02 |
| tp4_cp2 | forward | 0.2265 | 1.4769 | 6.5x | 0 | 0 | 1.033e-02 | nan |
| tp4_cp2 | train_fwd_bwd | 1.8629 | 8.3197 | 4.5x | 0 | 0 | 1.033e-02 | 1.099e-02 |
| tp4_cp2_sp | forward | 0.2949 | 2.1559 | 7.3x | 0 | 0 | 1.045e-02 | nan |
| tp4_cp2_sp | train_fwd_bwd | 1.9582 | 8.6712 | 4.4x | 0 | 0 | 1.045e-02 | 1.109e-02 |

## Communication overlap feasibility

The measured implementation deliberately serializes dependencies; the numbers above do not claim overlap. A representative 1 MiB deterministic collective costs 0.5396-1.0538 ms, while distributed Triton FFN forward costs 1.4769-2.5044 ms and forward+backward costs 5.1330-8.7284 ms.

- Forward SP all-gather feeds both gate/up GEMMs, and the final TP all-reduce or reduce-scatter consumes the down projection. These are hard data dependencies.
- Backward can overlap the fixed-order TP reduction of the gate contribution to dHidden with computation of the independent up contribution. The final wait and gate-then-up add order must remain unchanged to preserve 0 mismatch.
- CP gathers are prerequisites for weight-gradient GEMMs. Packing them in a fixed layout can amortize launch/signature overhead, but they cannot be hidden behind the GEMMs that consume them.
- Coalescing and host-side validation caching should be measured before introducing multi-stream scheduling complexity.

## Figures

![Single-GPU overhead](single_gpu_overhead.png)

![RCCL collective overhead](collective_overhead.png)

![Distributed FFN overhead](distributed_ffn_overhead.png)

![Accuracy trade-off](accuracy_tradeoff.png)
