# VIME Qwen3-8B TP2/CP2 200-round consistency experiment

This example measures train/rollout numerical consistency at two independent
layers: VIME's framework-level reuse of rollout log-probabilities and
RL-Kernel's operator-level alignment of Attention, dense FFN, and linear logp.
It is designed for one 8×H100 node with four Megatron training GPUs and four
vLLM rollout GPUs.

The optimization algorithm is explicitly fixed to GRPO with
`--advantage-estimator grpo`. DAPO-Math-17k is only the prompt/answer dataset;
it does not select the DAPO training algorithm. The rule reward is computed by
VIME's `deepscaler` reward implementation.

The experiment is fail-closed. A run is accepted only when its Ray job
succeeds, every expected operator route has CUDA execution readback, no
fallback or Triton route is observed for an R/R arm, the requested number of
steps is present, and vLLM CUDA Graph evidence matches the manifest.

## Ablation matrix

| Group | VIME `--use-rollout-logprobs` | Attention / FFN / logp | Purpose |
|---|---:|---|---|
| G00 | off | P/P | Production baseline |
| G10 | on | P/P | Framework-level consistency only |
| G01 | off | R/R | RL-Kernel operator-level consistency only |
| G11 | on | R/R | Framework-level plus operator-level consistency |

`P/P` selects the production implementation on training and rollout. `R/R`
selects RL-Kernel on both sides. All four groups use the same prompts, initial
checkpoint, sampling settings, seeds, TP2/CP2 topology, and batch sizes.

Do not interpret G10/G11 as evidence that train and rollout recomputation is
bitwise equal: framework reuse changes which stored logp enters the RL loss.
The direct numerical claim comes from G01/G11 and the runtime comparison
metrics.

## Required gates

- NVIDIA H100 × 8; actor GPUs 4; rollout GPUs 4; TP=2; CP=2; PP=1.
- GRPO, BF16, `top_p=1.0`, temperature 1, no dropout, fixed training and rollout seeds.
- A 4096-token response budget with full uniform activation recomputation
  (`recompute-num-layers=1`) and expandable CUDA allocator segments.
- vLLM CUDA Graph mode `FULL_DECODE_ONLY`, not eager, with exact capture sizes
  `1..(rollout_batch_size × n_samples_per_prompt)`.
- Megatron and vLLM readbacks for Attention, FFN, and logp, with positive call
  counts and the expected case/implementation.
- R/R runs must report zero bitwise mismatches, zero max absolute logp
  difference, CUDA execution, and no fallback or Triton provenance.
- Append-only run directories. A passing validator creates `COMPLETE`; failed
  attempts remain available for audit and are not overwritten.

The current VIME debug dump does not include training `log_probs` in
`rollout_data`. `validate_run.py` therefore uses VIME's runtime `torch.ne`,
maximum, and mean absolute-difference metrics. Counts are reconstructed from
the sample means and global batch size. The report marks offline tensor
comparison as unavailable instead of claiming it was performed.

## Recommended phases

The phase definitions are frozen in `experiment_matrix.json`.

| Phase | Steps | Seeds | Decision |
|---|---:|---|---|
| smoke | 1 | 1234 | Validate launch, routes, CUDA Graph, and artifacts |
| short | 8 | 1234 | Catch state transition, weight-update, and cache issues |
| precision | 30 | 1234, 2345, 3456 | Estimate drift distribution before the long run |
| convergence | 200 | 1234, 2345, 3456 | Primary PR evidence and learning/performance curves |

Use 200 steps for the main claim. One seed is enough to demonstrate a strict
bitwise invariant, but three paired seeds are recommended for reward,
throughput, and overhead claims. Run groups in the same seed order and compare
paired seeds; report the mean and a 95% confidence interval. Never merge runs
from different code revisions, checkpoints, prompt hashes, or CUDA Graph
settings in one estimate.

## Prepare DAPO-Math-17k

`prepare_dapo_data.py` downloads or converts the official Parquet file and
emits VIME `prompt`/`label` JSONL. It deduplicates by `extra_info.index` and
writes source/output hashes and row counts to a sibling manifest.

```bash
python examples/vime_qwen3_8b_tp2_cp2_200/prepare_dapo_data.py \
  --download \
  --source /data/dapo-math-17k.parquet \
  --output /data/dapo-math-17k.vime.jsonl
```

The converter requires `pyarrow`. The small
`qwen3_8b_multiround_math.jsonl` fixture is for smoke testing only and must not
be used for convergence or reward claims.

## Run one arm

Start a Ray cluster appropriate for the host, then invoke `run_arm.py`. The
example below is schematic; every path is recorded in `manifest.json`.

```bash
python examples/vime_qwen3_8b_tp2_cp2_200/run_arm.py \
  --group G01 \
  --num-rollout 8 \
  --seed 1234 \
  --rollout-seed 1234 \
  --output-root /data/vime-200/runs/short \
  --rl-kernel-root /path/to/RL-Kernel \
  --vime-root /path/to/vime \
  --megatron-root /path/to/Megatron-LM \
  --model-root /models/Qwen3-8B \
  --ref-load /models/Qwen3-8B_torch_dist \
  --prompt-data /data/dapo-math-17k.vime.jsonl \
  --python /path/to/python \
  --ray-bin /path/to/ray \
  --wait
```

`run_arm.py` refuses to reuse an existing run ID. It records repository
revisions, command line, environment, data hash, GPU inventory, topology,
seeds, batch parameters, and CUDA Graph contract before submission.

After the Ray job finishes, save its combined log as `run.log` in the run
directory and validate it:

```bash
python examples/vime_qwen3_8b_tp2_cp2_200/validate_run.py \
  --run-dir /data/vime-200/runs/short/<run-id> \
  --seal
```

## Aggregate and plot

Only sealed runs are collected. `collect_results.py` writes one row per run,
one row per training step, and group-level summaries.

```bash
python examples/vime_qwen3_8b_tp2_cp2_200/collect_results.py \
  --runs-root /data/vime-200/runs \
  --output-dir /data/vime-200/results

python examples/vime_qwen3_8b_tp2_cp2_200/plot_results.py \
  --rounds-csv /data/vime-200/results/rounds.csv \
  --phase convergence \
  --output-dir /data/vime-200/results/figures
```

The plotting step requires Matplotlib. It produces:

- `consistency.png`: mean/max absolute logp difference, mismatch rate, and
  mismatch count per step;
- `learning.png`: raw reward with moving average, PPO KL, entropy, and response
  truncation ratio;
- `performance.png`: end-to-end step time, rollout time, and actor throughput.

The summary table also reports total active-token exposure, cumulative
bitwise mismatch count, token-weighted mean absolute difference, maximum
absolute difference, reward, truncation, step time, and throughput. For R/R,
the strongest claim is `mismatch_count = 0` over the stated token exposure;
reward and speed are secondary quality and cost measurements.
