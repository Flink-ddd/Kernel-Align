# Vime Qwen3-8B TP=2 CP=2 validation

This example is the recommended reproducible entry point for the Vime-side
selected-logprob integration. It keeps framework glue in Vime and keeps the
numerical provider, contract, provenance, and report in RL-Kernel.

The example is deliberately strict:

- Megatron training uses `TP=2`, `CP=2`, `PP=1`, and four actor ranks.
- vLLM rollout uses processed logprobs and `top_p=1.0`.
- Vime must load `rl_engine.integrations.vime.linear_logp.provider` in `strict`
  mode. The former `rl_engine.integrations.vime.logp.provider` path remains a
  compatibility alias.
- A native fallback or a missing provider marker is not reported as a pass.
- Attention and FFN are not declared consistent from configuration alone. They
  require executed Megatron and vLLM readbacks, so the report marks them
  `unclaimed` until those artifacts are supplied. The readback must use
  `rlkernel.operator_runtime_evidence.v1` and report exact-zero comparison
  metrics for both sides.

The Vime companion must be installed or checked out separately. This example
does not modify `vllm-project/vime`.

## User modes

`RL_KERNEL_MODE` is the user-facing switch. The fair runner maps each mode to
one complete train/rollout route rather than asking users to compose module
cases manually:

| Mode | Route | Fallback policy | Intended use |
| --- | --- | --- | --- |
| `strict` | R/R | fail closed | production bitwise consistency |
| `audit` | R/R | record every fallback without a strict pass claim | diagnosis and route evidence |
| `auto` | P/P through installed adapters | observable | compatibility checks |
| `off` | native P/P, no provider or plugin injection | native by definition | clean baseline |

The recommended post-training comparison is `off` versus `strict`. Both use
the same aligned framework flags by default, so the measured delta is the
RL-Kernel backend cost rather than an unrelated determinism configuration.
The older `pp`, `pp-aligned`, `rr`, and `rr-aligned` names remain accepted.

```bash
examples/vime_qwen3_8b_tp2_cp2/run_fair_perf_case.sh off
examples/vime_qwen3_8b_tp2_cp2/run_fair_perf_case.sh strict
examples/vime_qwen3_8b_tp2_cp2/run_fair_perf_case.sh audit
```

The executable entry point is the Vime script
`scripts/run-qwen3-8B-rlkernel-tp2-cp2.sh`. Its default topology is an
8-GPU H100 node with four Megatron actor GPUs and four vLLM rollout GPUs. The
script refuses to start on a different GPU count or GPU class. Set
`COLOCATE=1` only when intentionally testing the colocated path; that mode is
not the default 8-GPU train/infer split.

## Dry run

```bash
python examples/vime_qwen3_8b_tp2_cp2/run.py \
  --vime-root /path/to/RL-Align/vime \
  --rl-kernel-root /path/to/RL-Kernel \
  --output reports/qwen3_8b_tp2_cp2.validation.json
```

## Execute

The Vime script expects model/checkpoint/data paths through environment
variables. Override them before adding `--run`:

```bash
export MODEL_ROOT=/models/Qwen3-8B
export TORCH_DIST_ROOT=/models/Qwen3-8B_torch_dist
export PROMPT_DATA=/data/dapo-math-17k.jsonl
export RL_KERNEL_ROOT=/path/to/RL-Kernel
export MEGATRON_ROOT=/path/to/Megatron-LM

python examples/vime_qwen3_8b_tp2_cp2/run.py \
  --vime-root /path/to/RL-Align/vime \
  --rl-kernel-root "$RL_KERNEL_ROOT" \
  --output reports/qwen3_8b_tp2_cp2.validation.json \
  --run
```

For a real 8xH100 run, the model, Megatron torch-dist checkpoint, prompt data,
and Megatron checkout must already exist on the host. The first run can omit
`VIME_CKPT`; the script will initialize from `TORCH_DIST_ROOT` and save the
Vime checkpoint there. Use `NUM_ROLLOUT=1` for the integration smoke test and
increase it only after the provider marker is observed.

When the Megatron/vLLM launch also emits the operator readback artifact, pass
it explicitly:

```bash
python examples/vime_qwen3_8b_tp2_cp2/run.py \
  --vime-root /path/to/RL-Align/vime \
  --rl-kernel-root "$RL_KERNEL_ROOT" \
  --runtime-evidence reports/qwen3_8b_tp2_cp2.runtime-evidence.json \
  --output reports/qwen3_8b_tp2_cp2.validation.json \
  --run
```

The evidence file is intentionally post-execution. It must include training
and rollout identities for `attention` and `ffn`, plus `passed: true` and
exact-zero `out`, backward, and (for attention) `LSE` comparison metrics. A
configured backend without this readback remains `unclaimed`.

The runner writes a JSON report and a sibling combined log. The report records
the exact command, both repository revisions, provider backend identity, strict
fallback status, and the claim boundary. It does not fabricate numerical drift
when the GPU run was not executed.
