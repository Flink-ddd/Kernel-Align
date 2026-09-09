# Vime ROCm Attention operator ablation

This example runs one real Vime rollout/training step for each Attention
implementation pairing:

| Case | Megatron training | vLLM rollout | Purpose |
| --- | --- | --- | --- |
| `P/P` | framework production | framework production | native baseline |
| `P/R` | framework production | RL-Kernel | rollout-side attribution |
| `R/P` | RL-Kernel | framework production | training-side attribution |
| `R/R` | RL-Kernel | RL-Kernel | strict ROCm control |

Only `RL_KERNEL_ATTENTION_CASE` changes. FFN and Logp stay on strict `R/R`; model,
reference checkpoint, prompt data, topology, batch shape, optimizer, seeds,
dropout, and vLLM execution mode are frozen. Megatron sequence parallelism is
explicitly disabled in every arm: the strict projection hook does not yet
implement the complete SP all-gather/reduce-scatter contract. Context
parallelism remains supported and is validated using Vime's zigzag shards.

Every arm explicitly selects `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA`. The
Megatron actor loads the same Hugging Face weights used to initialize vLLM,
pins `--start-rollout-id 0`, and never reuses another arm's output. The
Attention-only matrix does not instantiate a zero-coefficient reference model;
the reference checkpoint remains sealed as an input for later full-path runs.

`--get-mismatch-metrics` uses the included `metrics_only_tis` hook. The hook
returns the policy loss and masks unchanged—no TIS weighting or rejection is
introduced—and writes the rank/call evidence consumed by validation.

This is the executable Attention **operator cross-configuration matrix**. It
does not claim to execute PR #230's compact `A0`-`A7` diagnostic taxonomy.
Those rows describe one-at-a-time root-cause probes; most still need concrete
runtime mutation and restoration hooks. No row label is treated as execution
evidence here.

## HIP Graph execution

The launcher uses `FULL_AND_PIECEWISE` graph mode for every arm, with a maximum
capture size of 32. Stateful strict collectives remain at piecewise graph
boundaries. ROCm uses PyTorch's `torch.cuda` graph API to capture HIP work.
Attention routes and fixed-paged settings participate in the graph cache key.

## Prerequisites

Use a ROCm environment with Vime, Megatron-LM, vLLM, AITER, and this RL-Kernel
checkout installed (normally `pip install -e /work/RL-Kernel`). A source-only
`PYTHONPATH` entry is insufficient because vLLM discovers RL-Kernel through the
installed `vllm.general_plugins` entry point. The launcher is based on Vime's
Qwen3-8B AMD launcher and the existing `vime_qwen3_8b_tp2_cp2` example, but is
parameterized and avoids their CUDA-only flags.

The default formal topology reuses PR #377's colocated eight-GPU schedule:

- all eight GPUs host Megatron training with TP=4, CP=2, PP=1 and sequence parallelism disabled;
- the same eight GPUs host two vLLM TP=4 engines while rollout is active;
- the actor remains resident and rollout is offloaded between generation phases;
- the vLLM router is pinned to `round_robin`, with two independent prompt requests so both engines receive work;
- one rollout, 2 prompts, 1 sample per prompt, global batch 2;
- response length 32 and at most 256 packed tokens per training GPU;
- Qwen3-8B BF16 with deterministic seeds 1234/42.
- strict Logp vocabulary layout: 151936 real rows, padded to 152064 rows for TP4.

The batch defaults are deliberately small correctness settings for the strict
path and may be overridden consistently on the CLI. Supported strict rollout
calls consume the vLLM paged KV cache directly. Readbacks report
`dense_kv_materialized` so a materialized fallback can be distinguished from
this direct path. See [fixed paged CK attention](fixed_paged_ck.md) for the
opt-in arithmetic schedule, supported shapes, and one-round long-workload runner.

The command refuses to reuse a non-empty run directory or an already-running
Ray cluster. Each arm receives its own dump, readback, and log directory. A
final checkpoint is written only when `RLK_ABLATION_SAVE_CHECKPOINT=1`; the
matrix never loads another arm's output checkpoint. Before creating the
run directory, it also requires the reference checkpoint's
`latest_checkpointed_iteration.txt` marker and verifies that the installed
RL-Kernel distribution exposes the expected vLLM plugin entry point.

## Inspect the plan

No configuration JSON is checked into the repository. Supply paths on the CLI
or through the matching environment variables:

```bash
python examples/vime_rocm_attention_ablation/run.py \
  --vime-root /work/vime \
  --rl-kernel-root /work/RL-Kernel \
  --megatron-root /work/Megatron-LM-vime \
  --model-root /app/model/Qwen3-8B \
  --reference-checkpoint /app/model/Qwen3-8B_torch_dist \
  --prompt-data /app/model/dapo-math-17k/dapo-math-17k.jsonl
```

Without `--run`, this prints the exact four-arm plan and does not start Ray or
write results.

## Execute all four arms

```bash
python examples/vime_rocm_attention_ablation/run.py \
  --vime-root /work/vime \
  --rl-kernel-root /work/RL-Kernel \
  --megatron-root /work/Megatron-LM-vime \
  --model-root /app/model/Qwen3-8B \
  --reference-checkpoint /app/model/Qwen3-8B_torch_dist \
  --prompt-data /app/model/dapo-math-17k/dapo-math-17k.jsonl \
  --run-dir /work/RL-Kernel/runs/vime-rocm-attention-$(date -u +%Y%m%dT%H%M%SZ) \
  --run
```

The runner content-hashes the prompt dataset, launcher, and small checkpoint
index/config manifests. For the large model/checkpoint trees it seals every
relative file name, size, nanosecond mtime, and symlink target without rereading
all 8B weight shards. It also records each source revision, tracked dirty state,
and tracked-diff digest. Dirty checkouts are allowed, but the complete seal must
remain identical before and after the four arms.

The default resource arguments are equivalent to:

```bash
python examples/vime_rocm_attention_ablation/run.py \
  ... \
  --visible-gpus 0,1,2,3,4,5,6,7 \
  --num-gpus 8 \
  --tp-size 4 \
  --cp-size 2 \
  --rollout-tp-size 4 \
  --run
```

CP remains an Attention matrix dimension here; sequence parallelism remains
off even when CP is greater than one.

## Runtime artifacts and validation

Generated JSON is runtime evidence and lives only under `--run-dir` (the
repository's top-level `runs/` directory is ignored):

```text
<run-dir>/
  matrix-plan.json
  frozen-inputs.before.json
  frozen-inputs.after.json
  matrix-validation.json
  arms/
    p-p/
    p-r/
    r-p/
    r-r/
      launch.json
      launcher.log
      validation.json
      checkpoint/  # only when RLK_ABLATION_SAVE_CHECKPOINT=1
      dump/rollout_data/*.pt
      mismatch_sidecars/*.pt
      readbacks/
```

An arm passes only after execution proves the requested route on both sides:

- a Megatron/training and vLLM/rollout Attention hook was installed and called;
- `P` records production and does not resolve to RL-Kernel;
- `R` records ROCm execution, the strict AITER/CK Attention runtime/core and
  fixed no-Split-KV schedule, no fallback/reference path, and the approved
  deterministic `rlkernel.rocm.triton_det_gemm` QKV/O projection;
- the no-correction TIS hook atomically records rank/call-local training and
  rollout logprobs, full response masks, and sequence lengths as `.pt`
  sidecars; validation computes finite, non-empty mismatch count, max/mean
  absolute drift, forward mismatch KL, and K3 KL from that evidence;
- CP sidecars are interpreted with Vime's two-ended zigzag response slice and
  TP replicas are deduplicated. This observation path never depends on the
  transient `partition` field that Vime removes before saving train debug data;
- the `R/R` control is bitwise equal for training versus rollout logprobs.

The matrix additionally requires identical content fingerprints before/after
and identical rollout sample/token identity across all four arms. If Attention
changes generated tokens, the result remains useful operational evidence, but
the cross-arm metric comparison is marked invalid instead of comparing
different samples.

To revalidate a completed arm without launching Vime:

```bash
python examples/vime_rocm_attention_ablation/validate_artifacts.py \
  --arm-dir /path/to/run/arms/r-r \
  --case R/R
```

Do not add `matrix-plan.json`, validation JSON, rollout dumps, mismatch
sidecars, checkpoints, or MI300X result files to the PR. Publish them as CI/job
artifacts when needed.
