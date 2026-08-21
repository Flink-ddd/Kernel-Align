# PR #325 ROCm deterministic FFN performance analysis

> Operator-only benchmark. No model checkpoint or serving engine was used.

## Environment

| Item | Value |
|---|---|
| NCCL_IB_DISABLE | 1 |
| architecture | gfx942:sramecc+:xnack- |
| git_commit | 7d04efe80f7903431bc75fff3bbbe85eb82e02c4 |
| gpu | AMD Instinct MI300X |
| gpu_count | 8 |
| hip | 7.14.60850 |
| native_collective | torch.distributed ProcessGroupNCCL (RCCL on ROCm) |
| native_gemm | torch.matmul (ROCm rocBLAS/hipBLASLt dispatch) |
| python | 3.12.3 |
| torch | 2.12.0+rocm7.14.0a20260608 |

## Methodology

- BF16 operator inputs; FFN dimensions are H=4096 and I=12288.
- Native GEMM is `torch.matmul`; native communication calls PyTorch's ProcessGroupNCCL directly, which uses RCCL on ROCm.
- The native distributed FFN uses the same weights, TP/CP/SP sharding, and collective schedule as the deterministic path.
- Single-GPU timing: GPU events, median and p95; distributed timing: synchronized wall clock, slowest rank per sample.
- 5 warmups, 20 measured forward/collective samples, and 10 measured forward+backward samples.
- `NCCL_IB_DISABLE=1` selects the same intra-node XGMI transport for both collective implementations. Raw median, p95, minimum, and maximum measurements are in `results.json`.

Reproduce this report from the repository root:

```bash
python benchmarks/benchmark_rocm_ffn.py \
  --warmup 5 \
  --samples 20 \
  --training-samples 10 \
  --output-dir benchmarks/results/pr325_rocm_mi300x
```

## Key findings

- The gfx942 deterministic GEMM scalar fallback costs 44.0-1109.8x native latency. This dominates end-to-end FFN overhead.
- The fused deterministic SwiGLU costs 0.22-0.76x native latency, so it is faster than the unfused PyTorch expression; it is not the bottleneck.
- Deterministic collectives cost 2.0-27.6x native RCCL latency across 2/4/8 ranks and 64 KiB/1 MiB/16 MiB per rank.
- Distributed FFN overhead is 32.1-105.8x for forward and 11.7-61.0x for forward+backward.
- All 16 distributed direction/topology cases are train/inference and repeat bitwise consistent: yes.
- GEMM relative-L2 error against FP32 is 0.500-0.577% for the deterministic BF16 tree versus 0.164-0.166% for native ROCm. The fixed BF16 tree buys topology invariance, not better FP32 proximity.
- Distributed FFN output drift versus native is 1.012-1.060% relative-L2. Conversely, the balanced deterministic reduction is closer to FP32 than native RCCL in 12/12 tested 4/8-rank reduction cases.

## Single-GPU GEMM

| Shape | Native ROCm (ms) | Deterministic (ms) | Overhead | Det rel-L2 vs FP32 | Native rel-L2 vs FP32 |
|---|---:|---:|---:|---:|---:|
| gate_up_m1 | 0.0384 | 1.6887 | 44.0× | 5.003e-03 | 1.651e-03 |
| down_m1 | 0.0283 | 4.8950 | 173.2× | 5.769e-03 | 1.635e-03 |
| gate_up_m8 | 0.0297 | 3.1275 | 105.4× | 5.018e-03 | 1.663e-03 |
| down_m8 | 0.0299 | 6.3458 | 211.9× | 5.679e-03 | 1.654e-03 |
| gate_up_m32 | 0.0318 | 12.9507 | 406.9× | 5.019e-03 | 1.654e-03 |
| down_m32 | 0.0295 | 16.5279 | 560.6× | 5.604e-03 | 1.662e-03 |
| gate_up_m128 | 0.0533 | 45.9833 | 862.7× | 5.019e-03 | 1.657e-03 |
| down_m128 | 0.0550 | 61.0427 | 1109.8× | 5.623e-03 | 1.659e-03 |

## Single-GPU SwiGLU

| Case | Native PyTorch (ms) | Deterministic fused (ms) | Det/native | Det/native rel-L2 |
|---|---:|---:|---:|---:|
| swiglu_forward_m1 | 0.0168 | 0.0127 | 0.76x | 2.877e-03 |
| swiglu_backward_m1 | 0.0562 | 0.0141 | 0.25x | 5.159e-03 |
| swiglu_forward_m8 | 0.0190 | 0.0129 | 0.68x | 2.816e-03 |
| swiglu_backward_m8 | 0.0676 | 0.0146 | 0.22x | 5.074e-03 |
| swiglu_forward_m32 | 0.0200 | 0.0136 | 0.68x | 2.825e-03 |
| swiglu_backward_m32 | 0.0706 | 0.0156 | 0.22x | 5.071e-03 |
| swiglu_forward_m128 | 0.0203 | 0.0134 | 0.66x | 2.828e-03 |
| swiglu_backward_m128 | 0.0669 | 0.0149 | 0.22x | 5.059e-03 |

## Single-GPU FFN

| Case | Native ROCm (ms) | Deterministic (ms) | Overhead | Output rel-L2 | Max grad rel-L2 |
|---|---:|---:|---:|---:|---:|
| ffn_forward_m1 | 0.0929 | 9.4786 | 102.0× | 1.008e-02 | nan |
| ffn_train_fwd_bwd_m1 | 0.4308 | 39.0465 | 90.6× | 1.008e-02 | 1.037e-02 |
| ffn_forward_m8 | 0.1018 | 14.0304 | 137.8× | 9.930e-03 | nan |
| ffn_train_fwd_bwd_m8 | 0.4067 | 47.4899 | 116.8× | 9.930e-03 | 1.017e-02 |
| ffn_forward_m32 | 0.1120 | 43.5693 | 388.9× | 9.952e-03 | nan |
| ffn_train_fwd_bwd_m32 | 0.5347 | 107.4444 | 200.9× | 9.952e-03 | 1.024e-02 |

## RCCL collectives

| Operation | Ranks | Input/rank | Native RCCL (ms) | Deterministic (ms) | Overhead | Det/native rel-L2 | Native/FP32 rel-L2 | Det/FP32 rel-L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_reduce | 2 | 0.0625 MiB | 0.0592 | 0.5532 | 9.3× | 0.000e+00 | 1.798e-03 | 1.798e-03 |
| all_gather | 2 | 0.0625 MiB | 0.0534 | 0.5392 | 10.1× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 2 | 0.0625 MiB | 0.0523 | 0.5616 | 10.7× | 0.000e+00 | 1.804e-03 | 1.804e-03 |
| all_reduce | 2 | 1 MiB | 0.0825 | 0.5553 | 6.7× | 0.000e+00 | 1.800e-03 | 1.800e-03 |
| all_gather | 2 | 1 MiB | 0.0772 | 0.5586 | 7.2× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 2 | 1 MiB | 0.0719 | 0.5781 | 8.0× | 0.000e+00 | 1.803e-03 | 1.803e-03 |
| all_reduce | 2 | 16 MiB | 0.5073 | 1.0212 | 2.0× | 0.000e+00 | 1.800e-03 | 1.800e-03 |
| all_gather | 2 | 16 MiB | 0.4099 | 1.2952 | 3.2× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 2 | 16 MiB | 0.2442 | 0.8913 | 3.6× | 0.000e+00 | 1.803e-03 | 1.803e-03 |
| all_reduce | 4 | 0.0625 MiB | 0.0617 | 0.7073 | 11.5× | 3.221e-03 | 2.674e-03 | 2.536e-03 |
| all_gather | 4 | 0.0625 MiB | 0.0573 | 0.6558 | 11.4× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 4 | 0.0625 MiB | 0.0559 | 0.7087 | 12.7× | 3.263e-03 | 2.734e-03 | 2.559e-03 |
| all_reduce | 4 | 1 MiB | 0.0874 | 0.6935 | 7.9× | 3.256e-03 | 2.682e-03 | 2.566e-03 |
| all_gather | 4 | 1 MiB | 0.0883 | 0.6794 | 7.7× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 4 | 1 MiB | 0.0908 | 0.7014 | 7.7× | 3.319e-03 | 2.691e-03 | 2.599e-03 |
| all_reduce | 4 | 16 MiB | 0.2442 | 1.0390 | 4.3× | 3.256e-03 | 2.682e-03 | 2.566e-03 |
| all_gather | 4 | 16 MiB | 0.4244 | 1.0259 | 2.4× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 4 | 16 MiB | 0.1556 | 1.0519 | 6.8× | 3.319e-03 | 2.691e-03 | 2.599e-03 |
| all_reduce | 8 | 0.0625 MiB | 0.0374 | 1.0311 | 27.6× | 4.446e-03 | 3.664e-03 | 3.099e-03 |
| all_gather | 8 | 0.0625 MiB | 0.0721 | 0.9555 | 13.2× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 8 | 0.0625 MiB | 0.0670 | 1.0387 | 15.5× | 4.879e-03 | 3.780e-03 | 3.155e-03 |
| all_reduce | 8 | 1 MiB | 0.0545 | 1.0036 | 18.4× | 4.474e-03 | 3.676e-03 | 3.110e-03 |
| all_gather | 8 | 1 MiB | 0.0887 | 0.9689 | 10.9× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 8 | 1 MiB | 0.0669 | 1.0162 | 15.2× | 4.765e-03 | 3.768e-03 | 3.168e-03 |
| all_reduce | 8 | 16 MiB | 0.1812 | 1.3730 | 7.6× | 4.474e-03 | 3.676e-03 | 3.110e-03 |
| all_gather | 8 | 16 MiB | 0.4353 | 1.3738 | 3.2× | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| reduce_scatter | 8 | 16 MiB | 0.1228 | 1.5048 | 12.3× | 4.765e-03 | 3.768e-03 | 3.168e-03 |

## Distributed FFN

| Topology | Direction | Native ROCm/RCCL (ms) | Deterministic (ms) | Overhead | Output rel-L2 | Max grad rel-L2 | Train/infer bitwise | Repeat bitwise |
|---|---|---:|---:|---:|---:|---:|:---:|:---:|
| tp2 | forward | 0.2170 | 22.9695 | 105.8× | 1.012e-02 | nan | yes | yes |
| tp2 | train_fwd_bwd | 0.9359 | 57.1264 | 61.0× | 1.012e-02 | 1.079e-02 | yes | yes |
| tp2_sp | forward | 0.2821 | 23.6457 | 83.8× | 1.015e-02 | nan | yes | yes |
| tp2_sp | train_fwd_bwd | 1.0045 | 58.6581 | 58.4× | 1.015e-02 | 1.083e-02 | yes | yes |
| tp4 | forward | 0.2175 | 13.6428 | 62.7× | 1.031e-02 | nan | yes | yes |
| tp4 | train_fwd_bwd | 1.1884 | 33.2462 | 28.0× | 1.031e-02 | 1.095e-02 | yes | yes |
| tp2_cp2 | forward | 0.2231 | 14.1083 | 63.2× | 1.015e-02 | nan | yes | yes |
| tp2_cp2 | train_fwd_bwd | 1.2568 | 43.0444 | 34.2× | 1.015e-02 | 1.083e-02 | yes | yes |
| tp2_cp2_sp | forward | 0.2819 | 14.7152 | 52.2× | 1.022e-02 | nan | yes | yes |
| tp2_cp2_sp | train_fwd_bwd | 1.3499 | 44.1678 | 32.7× | 1.022e-02 | 1.089e-02 | yes | yes |
| tp8 | forward | 0.1939 | 8.2995 | 42.8× | 1.060e-02 | nan | yes | yes |
| tp8 | train_fwd_bwd | 1.0345 | 20.4734 | 19.8× | 1.060e-02 | 1.126e-02 | yes | yes |
| tp4_cp2 | forward | 0.2230 | 8.4606 | 37.9× | 1.033e-02 | nan | yes | yes |
| tp4_cp2 | train_fwd_bwd | 2.1949 | 25.7398 | 11.7× | 1.033e-02 | 1.099e-02 | yes | yes |
| tp4_cp2_sp | forward | 0.2846 | 9.1342 | 32.1× | 1.045e-02 | nan | yes | yes |
| tp4_cp2_sp | train_fwd_bwd | 2.0314 | 26.6942 | 13.1× | 1.045e-02 | 1.109e-02 | yes | yes |

## Figures

![Single-GPU overhead](single_gpu_overhead.png)

![RCCL collective overhead](collective_overhead.png)

![Distributed FFN overhead](distributed_ffn_overhead.png)

![Accuracy trade-off](accuracy_tradeoff.png)
