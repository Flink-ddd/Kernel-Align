# P1 mHC + RMSNorm Start Kit (`P1-S0`)

The start kit unblocks every P1 sub-issue (`P1-D1`…`P1-D6`): it freezes the
layer contract, provides a bit-exact FP32 oracle for the six WS1 operators,
generates seeded golden fixtures, and ships one acceptance command that any
backend PR can run independently.

Issue: [DSV4][P1/7] mHC 与 RMSNorm 确定性前向/反向 (#2).

## Sub-issue naming

| Label | Scope |
| --- | --- |
| `P1-S0` | This start kit (contract, oracle, fixtures, acceptance command) |
| `P1-D1` | `hc_split_sinkhorn` (fwd + bwd through all 20 Sinkhorn rounds) |
| `P1-D2` | `fp32_gemm_rms` (controller projection + controller RMS, fwd + bwd) |
| `P1-D3` | `mhc_post` (fwd + `dR_old`/`dy`/`dC`/`dPOST`) |
| `P1-D4` | `mhc_pre` / `h_aggregate` + the TE/Megatron/Miles provider adapter |
| `P1-D5` | `rmsnorm_residual` (fwd + `dX`/`dGamma`) |
| `P1-D6` | Fixed-K / batch-invariant GEMM reference + equivalence harness |
| `P1-R0` | Review, package the P1 provider artifact, GPU CI |

## What is in the kit

| Module | Contents |
| --- | --- |
| `rl_engine/mhc/reduction.py` | The two pinned reduction trees. Defines the golden bytes for every P1 accumulation. |
| `rl_engine/mhc/contract.py` | `LayerContract`, `ResidualBatch`, `ControllerParams`, `NormParams`, `GradBoundary`, fingerprints. |
| `rl_engine/mhc/oracle.py` | FP32 reference for the six operators plus the full block forward/backward composition. |
| `rl_engine/mhc/provider.py` | `MHCProvider` protocol, `ReferenceProvider` (oracle-backed), `StubProvider` (fail-closed), `check_capability`. |
| `rl_engine/mhc/fixtures.py` | Seeded fixture cases and the golden-hash manifest (`tests/fixtures/p1/golden_hashes.json`, the CI anchor). |
| `rl_engine/mhc/trace.py` | Boundary hashes + `first_divergence` (P1-local stand-in for `TraceEnvelope`). |
| `scripts/check_p1.py` | The acceptance command. |

## The reduction trees (the heart of the contract)

Everything in `oracle.py` reduces through `reduction.py`; no `torch.sum`,
`matmul`, `mean` or `einsum` appears anywhere in the operator bodies. Two
trees, and only two:

- **Long reductions** — the `K = 4·D` controller dot, the `D`-wide
  sum-of-squares, the token-major parameter gradients — use a **single FP32
  accumulator walking ascending indices left to right**, with every multiply
  and add rounding separately. This is the order the repository's existing
  `reduce_rows_fp32` left fold already uses, so a P1 kernel and the WS1 VJP
  path agree by construction.
- **4-element stream reductions** — the four mHC residual streams, and the
  row/column sums of the 4×4 Sinkhorn matrix — use the balanced tree
  **`(a0+a1)+(a2+a3)`** pinned by #2.

Banned downstream: Split-K, Stream-K, atomic partial accumulation, and any
order that varies with batch size, token count, SM count or any other runtime
condition. `tests/test_p1_reduction.py` pins both trees with cases where the
alternatives visibly disagree.

## Frozen numeric contract (recap of #2 + decisions made here)

From the issue:

1. All multiplies and reduction accumulations are FP32.
2. Each operator performs **exactly one** FP32→BF16 downcast, at its output:
   `mhc_pre`'s aggregated hidden, `rmsnorm_residual`'s normalized row, and
   `mhc_post`'s `R_new`. No intermediate BF16 cast anywhere.
3. `PRE`, `POST`, `C`, the controller projection `P` and the RMS scale `r`
   stay FP32 while travelling between operators.
4. Controller arithmetic is `h = ((r * P) * alpha) + bias`;
   `PRE[i] = sigmoid(h[i]) + 1e-6`; `POST[i] = 2·sigmoid(h[4+i])`;
   `L = h[8:24].reshape(4,4)`.
5. `hc_mult = 4`, layout `PRE[0:4] POST[4:8] COMB[8:24]`, `sinkhorn_iters = 20`,
   `eps = 1e-6`; `sum + eps` may not be replaced by a `clamp`.
6. `rmsnorm_residual` is `rsqrt(mean(x²) + eps)`; the controller RMS is
   `1/(sqrt(mean(x²)) + eps)`. The two forms are never interchangeable.

Decisions this kit had to freeze (flagged for review on #2; changing any of
them means regenerating the manifest and bumping the schema/profile id):

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | **`q = sqrt(s) / sqrt(K)`, not `sqrt(s / K)`.** The issue's forward text says `norm/sqrt(16384)` while its backward text says `q = sqrt(s/N)`; those differ in FP32. The forward wording wins, and backward reuses the saved `q`. | The two forms are not bit-equal; the forward is the authoritative statement. **Please confirm in review.** |
| D2 | **Sinkhorn `sum_row(M)[i] = Σ_j M[i,j]` (row sums, broadcast along j) and `sum_col(M)[j] = Σ_i M[i,j]`.** Schedule is literal: `softmax_row(L)+eps`, one column normalize, then 19×(row, column) — 20 column normalizations, 39 in total. | The issue names the steps but not the axis convention; this is the reading that makes rows/columns each sum to 1. |
| D3 | **`softmax_row` subtracts the row max before `exp`**, with the max taken on the same balanced 4-way tree. | Unshifted `exp` overflows on the saturating-logit fixture; the shift has to be pinned rather than left to the kernel. |
| D4 | **Sinkhorn backward walks the recorded 39 normalizations in reverse, one VJP per step.** No fixed-point / implicit-differentiation shortcut, no fused simplification. | #2 forbids "mathematically equivalent but differently associated" forms. Cross-checked against autograd in `test_p1_oracle.py`. |
| D5 | **Numeric profile `oracle-fp32-mhc-v1`**: FP32, the two trees above, mul-then-add (**no FMA fusion**). A strict CUDA kernel matches with `__fmul_rn`/`__fadd_rn` or registers its own profile. | Byte-equality needs the rounding points pinned, not just the order. |
| D6 | **`alpha` and `bias` are FP32 `[24]` vectors**, applied per controller output. | The general case; a scalar `alpha` is the broadcast special case and stays representable. |
| D7 | **The transformer sublayer is external to P1.** `y_sublayer` enters the block as data and `d_normalized` / `d_residual` enter the backward as data; `dy_sublayer` is an output boundary. | This is what makes P1 acceptance runnable with no P2–P7 code in the loop, exactly as the issue requires. |
| D8 | **The fused pre+norm boundary is defined as the unfused composition**, byte for byte. | Miles/XoRL fuse pre-mix and normalize into one launch. Defining the fused boundary this way turns "fused equals unfused" into a test rather than an assumption; a kernel whose fused residual store changes the reduction layout must register a different profile instead of presenting itself as the same kernel. |
| D9 | **`trainability='mixer-frozen'` returns `None` for `d_controller_weight`/`d_alpha`/`d_bias`**, not zeros. | #2: a stop-grad mixer must not leak `dMixWeight`. `None` cannot be silently summed into an optimizer; a zero tensor can. |
| D10 | **Gradients are returned FP32** (the accumulator dtype); rounding to BF16 happens only at an outer block edge. | Consistent with "FP32 reductions, BF16 boundaries". |

## Byte-equality scope

Strict byte-equality is required **between Megatron training and Miles
inference on the same numeric profile and device**. The committed manifest
anchors the CPU x86 oracle; `scripts/check_p1.py` recomputes the oracle on the
provider's device, so transcendentals (`sigmoid`, `exp`, `rsqrt`) never cross
devices inside a strict comparison. Hardware without equivalent capability
must register its own profile with an explicit tolerance — never silently
relax.

The acceptance command additionally re-runs each fixture row on its own and
checks that the bytes do not move: **same row, different batch / padding /
stride ⇒ identical output**, which is acceptance criterion 2 of #2.

## How a sub-issue PR uses the kit

1. Subclass `ReferenceProvider`, override only the operators your PR delivers
   (everything else stays on the oracle), and set `name` / `numeric_profile`:

   ```python
   from rl_engine.mhc.provider import ReferenceProvider

   class MyCudaProvider(ReferenceProvider):
       name = "my-cuda"
       numeric_profile = "cuda-ffma-strict-v1"

       def mhc_post_fwd(self, r_old, y, c, post):
           return my_cuda_kernel(r_old, y, c, post)
   ```

2. Run `python scripts/check_p1.py --provider your.module:YourProvider
   [--device cuda]`. Every boundary must be byte-equal; exit code 1 otherwise.
3. Ship the check output and your `provenance()` in the PR description.

Fixture cases: `one_row`, `packed_t16`, `packed_t7_odd`, `fused_pre_norm`,
`mixer_frozen`, plus operator edge cases `sinkhorn_edges` (saturating
sigmoids, tied logits, a zero row, and a magnitude that makes the `sum + eps`
guard load-bearing) and `rms_edges` (zero row, subnormal-ish and large
magnitudes, exact powers of two).

Fixture geometry is a scaled-down layer (`hidden = 128` ⇒ `K = 512`) so the
serial oracle stays CPU-cheap. `hc_mult`, `controller_n`, `sinkhorn_iters` and
both epsilons are the real production constants; the full DSv4 geometry
(`hidden = 4096`, `K = 16384`, `N = 24`) is pinned separately by
`LayerContract.assert_production()`.

Regenerate the manifest after an intentional contract change:

```bash
python -m rl_engine.mhc.fixtures --write-manifest
```

## Open questions for review

- **D1** — `sqrt(s)/sqrt(K)` vs `sqrt(s/K)` in the controller RMS. The issue
  text says both; the kit takes the forward wording. One line of Miles/Megatron
  source settles it.
- **Sinkhorn iteration detail** — #2 notes that TogetherAI never published the
  per-round detail and that the checkpoint plus the Miles implementation are
  the only reference. The kit implements the schedule exactly as written in
  the issue; if Miles differs, D2/D4 change and the manifest is regenerated.
- **D8** — whether Miles can expose the pre-fusion intermediate at all. If it
  cannot, `fused-pre-norm` becomes the only boundary for the inference side
  and `unfused` stays the training-side definition, with the equivalence case
  as the bridge.

## Non-goals of the kit

No CUDA/Triton kernels, no TE/Megatron/vLLM injection, no RoPE (that is P2's,
per #1), no attention or MoE (#4, #8/#10), and no WS2 TP/SP/CP/PP gates. The
`placement` field and the WS2 notes in the operator docstrings mark where those
gates will attach; `check_capability` already fails closed on any placement a
provider has not declared.
