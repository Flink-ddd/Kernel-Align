<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 RL-Kernel Contributors -->

# `rl_engine/mismatch` — detecting and reporting training-inference mismatch

The rollout policy `π_old` and the training policy `π_θ` compute logprobs for the
**same tokens with the same weights** and still disagree. This package turns
"which of the dozens of possible causes is it" into named **factors**, each a
switch that can be flipped one at a time and attributed to a side.

The output is a report: which factors were measured, which could not be and why,
and the few most suspicious modules ranked.

## Why the tail, not the mean

`dlogp = log π_θ − log π_old` enters the GRPO objective through `ρ = exp(dlogp)`,
which the objective clips at `1 ± ε`. With `ε = 0.2`, any token past
`|dlogp| > ln(1.2) ≈ 0.182` has its **gradient signal discarded** — and not
random tokens, the most mismatched ones.

A healthy `dlogp_mean` is `0.002–0.008` (dense) or `0.01–0.03` (large MoE), all
far below that edge. **Judging on the mean alone always concludes "everything is
fine."** Hence `dlogp_p99` / `dlogp_max` / `clip_fraction` / `worst_token`, and a
diagnosis matrix that converges on `clip_fraction`.

## Four arms, then four gates

A factor expands into four arms, not on/off — only a one-sided swap identifies a
side, and only a two-sided swap proves the reference itself is sound:

| arm | rollout | training | what it buys |
|---|---|---|---|
| `both_native` | native | native | the baseline the others are measured against |
| `both_reference` | reference | reference | **self-check gate** — must be bitwise identical, or this factor's conclusions are void |
| `training_reference_only` | native | reference | deviation gone ⇒ training side is the source |
| `rollout_reference_only` | reference | native | deviation gone ⇒ rollout side is the source |

A factor with no reference implementation is a **parameter sweep** instead: one
arm per allowed value. A sweep measures; it cannot conclude, because nothing was
swapped and so there is no side to attribute to.

Before any verdict, four gates run. **"Not measured" and "measured and clean"
are different things**, and confusing them is the mistake an attribution
framework is most likely to make:

| gate | fails when | verdict |
|---|---|---|
| 1 · did it apply | any arm is not `APPLIED` | `VARIANT_DID_NOT_APPLY`, with the resolution trace |
| 2 · evidence | `required_evidence` incomplete | `INSUFFICIENT_EVIDENCE` |
| 3 · shards | fewer logprob shards than `world_size` | `INSUFFICIENT_EVIDENCE` |
| 4 · guards | a pitfall guard failed | `INSUFFICIENT_EVIDENCE` |

`SwitchStatus.FELL_BACK` is why gate 1 exists: the reference was requested, the
engine silently reverted to native, and "the deviation did not change" then reads
as a clean `NOT_THIS_FACTOR`.

## Noise floors

A result only means something at a floor that can resolve it. Each step down
adds exactly one new noise source, so a failure points at a known suspect set. A
floor that has not passed blocks the next.

| floor | configuration | new noise source |
|---|---|---|
| `SINGLE_LAYER_ANCHOR` | 1 layer, single device, determinism on | none — failing bitwise here is an **operator bug**, not mismatch |
| `FULL_MODEL_SINGLE_GPU` | all layers, single device | accumulation over depth |
| `SHARDED_SINGLE_NODE` | TP + SP on one node | reduction order — the first floor with *real* mismatch |
| `PRODUCTION` | target TP/CP/PP, determinism off, decode | everything else; the only floor readable against `EXPECTED_RANGES` |

## Layout

```
mismatch/
├── schema/              pure data types, frozen, no behaviour
├── pipeline/            registry → planner → runner → diagnosis → report
├── engines/             the two sides under test: megatron.py, vllm.py
├── reference_adapters/  delivering pinned settings, and reading them back
├── model_meta/          per-model correspondence and call chain (qwen3.py)
├── operator_checks/     plugins, one directory per operator
├── docs/                tutorials
└── __main__.py          CLI, and the only module that imports plugins
```

`engines/` holds **`megatron.py` and `vllm.py` and nothing else** — the two
policies as they really run, shared across operators. Anything that merely
satisfies `ScoringBackend` is a harness, not a side under test, and lives in
`tests/` (see `tests/mismatch_cpu_backend.py`). The line is role, not protocol.

Three dependency rules keep the plugin seam open:

1. `schema/` never imports `pipeline/`; inside `schema/`, `values.py` imports
   nothing from the project.
2. `pipeline/` never imports `operator_checks/` — it sees only what the registry
   hands it. Break this and adding an operator becomes changing the framework.
3. Only `__main__` imports `operator_checks/`, which triggers registration.

## Running it

```bash
python -m rl_engine.mismatch list                    # operators and their factors
python -m rl_engine.mismatch plan --gpu-count 2      # expand into cases, cheapest first
python -m rl_engine.mismatch plan --json
```

Every `adapter.py` currently raises `NotImplementedError`, but the declaration
layer works — so a factor can be checked as wired before anything is implemented.

## Adding to this package

**Adding an operator is adding a directory; adding a factor is adding a file.**
No existing file changes, apart from one line in `__main__._OPERATOR_PACKAGES`
for a new operator. If your change needs an edit inside `pipeline/`, a global
dict, or another operator's directory, the framework is missing an abstraction —
raise it rather than patching around it.

| you want to | read |
|---|---|
| add a kernel's factor | [`docs/add-a-kernel-factor.md`](docs/add-a-kernel-factor.md) |
| add a communication feature | [`docs/add-a-comm-feature.md`](docs/add-a-comm-feature.md) |
