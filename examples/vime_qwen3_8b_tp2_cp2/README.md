# Vime Qwen3-8B TP=2 CP=2 validation

This example is the recommended reproducible entry point for the Vime-side
selected-logprob integration. It keeps framework glue in Vime and keeps the
numerical provider, contract, provenance, and report in RL-Kernel.

The example is deliberately strict:

- Megatron training uses `TP=2`, `CP=2`, `PP=1`, and four actor ranks.
- vLLM rollout uses processed logprobs and `top_p=1.0`.
- Vime must load `rl_engine.integrations.vime.logp.provider` in `strict` mode.
- A native fallback or a missing provider marker is not reported as a pass.
- Attention and FFN are not declared consistent from configuration alone. They
  require executed Megatron and vLLM readbacks, so the report marks them
  `unclaimed` until those artifacts are supplied. The readback must use
  `rlkernel.operator_runtime_evidence.v1` and report exact-zero comparison
  metrics for both sides.

The Vime companion must be installed or checked out separately. This example
does not modify `vllm-project/vime`.

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

python examples/vime_qwen3_8b_tp2_cp2/run.py \
  --vime-root /path/to/RL-Align/vime \
  --rl-kernel-root "$RL_KERNEL_ROOT" \
  --output reports/qwen3_8b_tp2_cp2.validation.json \
  --run
```

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
