# PR230 Attention taxonomy: ROCm operator micro-probes

> This applies PR230's row taxonomy to deterministic operator micro-probes.
> It is not the frozen model/rollout replay from PR230: no checkpoint, token stream,
> selected-token logprob, KL, serving engine, or AITER production claim is included.

## Environment

- GPU: AMD Instinct MI300X VF (gfx942:sramecc+:xnack-)
- PyTorch: 2.12.0+rocm7.14.0a20260608; HIP: 7.14.60850
- RL-Kernel: `2ea63b22b74feb6e5a748d09780fa075d5e644ed`
- Shapes: B=1, S=16, B=1, S=32, B=1, S=64, B=1, S=128, B=2, S=16, B=2, S=32, B=2, S=64, B=2, S=128
- Primary core: `rlkernel.rocm.deterministic_attention`

## Matrix

| Row | Factor | Comparable | Out | LSE | dQ | dK | dV | Result |
|---|---|:---:|---:|---:|---:|---:|---:|:---:|
| A0 | Strict replay baseline | yes | `0` | `0` | `0` | `0` | `0` | **PASS** |
| A1 | Position / RoPE | yes | `0.7109375` | `0.39382553` | `1.4140625` | `2.234375` | `0.78320312` | **PASS** |
| A2 | Q/K preprocessing | yes | `0.58203125` | `1.0303755` | `1.015625` | `1.7578125` | `0.96875` | **PASS** |
| A3 | Mask / sequence boundary | yes | `4.21875` | `7.671814` | `4.1968994` | `6.15625` | `10.984375` | **PASS** |
| A4 | Topology / head ownership | no | — | — | — | — | — | **REJECTED** |
| A5 | KV-cache identity / layout | yes | `6` | `4.578392` | `4.2773438` | `6.6274414` | `13.233398` | **PASS** |
| A6 | Numerical policy | yes | `0.1171875` | `0.099507809` | `0.09375` | `0.09375` | `0.11328125` | **PASS** |
| A7 | Distributed schedule | yes | `0.001953125` | `9.5367432e-07` | `0.00390625` | `0.00390625` | `0.00390625` | **PASS** |
| C0 | Invariant control | yes | `0` | `0` | `0` | `0` | `0` | **PASS** |
| C1 | Invariant control | yes | `0` | `0` | `0` | `0` | `0` | **PASS** |
| C2 | Invariant control | yes | `0` | `0` | `0` | `0` | `0` | **PASS** |

## Probe realizations

- `A0` — Repeat the identical native HIP reference-core call.
- `A1` — Increment suffix RoPE positions while preserving Q/K/V tensors.
- `A2` — Apply or bypass unit-weight PyTorch RMSNorm before the native HIP core.
- `A3` — Toggle causal masking in the native HIP core.
- `A4` — Bind valid TP-rank-1 rollout and TP-rank-0 training contracts; do not run numerics.
- `A5` — Reverse four dense K/V tensor pages; this is not a paged-cache runtime.
- `A6` — Use identical FP32 products/order with explicit FP32 versus BF16 accumulator state.
- `A7` — Merge four dense chunks in opposite orders on one GPU; this is not a CP collective.
- `C0` — Compare full GQA with two contiguous TP=2 head shards on one GPU.
- `C1` — Compare a batch call with per-row calls to the same native HIP core.
- `C2` — Compare a full-prefill tail with one trailing query over dense KV, without a serving cache.

A1-A3 and A5-A7 deliberately inject one mismatch and report the worst max-absolute
difference over the shape sweep. A4 is rejected by the repository's cross-config
binding gate before numerical comparison. A0 and C0-C2 must be bitwise zero for
Out/LSE/dQ/dK/dV.

A6 and A7 are eager PyTorch-on-ROCm probes for accumulation and merge order; the
remaining numerical rows invoke the native deterministic HIP Attention core. This
is operator-only reference evidence, not full PR230 replay evidence.

The complete per-shape mismatch counts and max-absolute values are in `results.json`.

## Reproduce

Run from the recorded clean commit and choose a new output directory:

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python \
  benchmarks/benchmark_rocm_attention_ablation.py --device 0 \
  --output-dir /tmp/pr230_rocm_mi300x_ablation
```
