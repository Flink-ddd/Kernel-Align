<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 RL-Kernel Contributors -->

# `rl_engine/mismatch` — detecting and reporting training-inference mismatch

The rollout policy `π_old` and the training policy `π_θ` compute logprobs for the
**same tokens with the same weights** and still disagree. This package turns
"which of the dozens of possible causes is it" into a set of named **factors**,
each of which is a switch that can be flipped one at a time, run, and attributed
to a side.

The output is not a number. It is a **report**: which factors were measured,
which could not be measured and why, and the few most suspicious modules ranked
by suspicion.

## Why the mismatch matters

`dlogp = log π_θ − log π_old` enters the GRPO/PPO objective through the
importance ratio `ρ = exp(dlogp)`, which the objective clips at `1 ± ε`. With the
usual `ε = 0.2`, any token past `|dlogp| > ln(1.2) ≈ 0.182` has its **gradient
signal discarded** — and the discarded tokens are not random, they are the ones
with the largest mismatch.

The trap: at the production floor a healthy `dlogp_mean` is `0.002–0.008` (dense)
or `0.01–0.03` (large MoE), all far below the clip edge. **Judging on the mean
alone always concludes "everything is fine."** The danger lives in the tail, which
is why `MismatchMetrics` carries `dlogp_p99` / `dlogp_max` / `clip_fraction` /
`approx_kl` / `worst_token`, and why the diagnosis matrix converges on
`clip_fraction`, not on the mean (`pipeline/diagnosis.py::_converged`).

## What it does, concretely

1. **Collects factors** from operator plugins at import time, rejecting duplicate
   ids, duplicate switch paths, and one contract field claimed at two different
   comparison rules.
2. **Rejects self-contradictory declarations statically** — claiming topology
   independence while reducing in NCCL's chosen order produces numbers that mean
   nothing, and that is knowable before anything runs.
3. **Expands each factor into four arms**, not on/off:

   | arm | rollout | training | what it buys |
   |---|---|---|---|
   | `both_native` | native | native | the baseline every other arm is measured against |
   | `both_reference` | reference | reference | **self-check gate** — must be bitwise identical, or every conclusion from this factor is void |
   | `training_reference_only` | native | reference | deviation gone ⇒ the training side is the source |
   | `rollout_reference_only` | reference | native | deviation gone ⇒ the rollout side is the source |

   A factor with no reference implementation is a **parameter sweep** instead:
   one arm per allowed value.
4. **Orders cases by rebuild cost** so a run reuses engines instead of restarting
   the process ~160 times.
5. **Runs four gates before any verdict** — "not measured" and "measured and
   clean" are different, and confusing them is the mistake an attribution
   framework is most likely to make.
6. **Diagnoses** each factor, then **traces root causes** across all of them into
   a ranked `MismatchReport`.

## The five types that carry the whole design

| type | what it is |
|---|---|
| `MismatchFactor` | one suspected cause — a switch, its comparison rules, its prerequisites, its pitfalls |
| `FactorVariant` | one ablation arm of that factor, as pasteable switch values |
| `VariantResult` | what one arm produced: status, metrics, evidence, effective config |
| `Diagnosis` | the verdict for one factor: training side, rollout side, both, not this factor, or *cannot tell* |
| `OperatorChecks` | the plugin protocol — one operator's factors plus how to read that operator back from an engine |

One factor expands into arms, each arm produces a result, the results become one
diagnosis, and all diagnoses become one report. Everything else is detail.

## The pipeline, in order

```
1  registry     collect plugins and factors           pipeline/registry.py
2  planner      filter by prerequisites               pipeline/planner.py
2.5 planner     reject contradictory declarations
3  planner      expand factors into variants
4  planner      order cases by rebind cost
5  runner       run each arm on both sides            pipeline/runner.py
              → contracts compared field by field    pipeline/comparison.py
              → dlogp / ρ / clip_fraction
6  diagnosis    four gates, then the matrix           pipeline/diagnosis.py
7  report       filter false positives, rank causes   pipeline/report.py
```

The four gates in step 6, in order — none of them may be skipped:

| gate | fails when | verdict |
|---|---|---|
| 1 · did it apply | any arm is not `APPLIED` | `VARIANT_DID_NOT_APPLY` (with the resolution trace: what was tried, why rejected) |
| 2 · evidence | `required_evidence` is incomplete | `INSUFFICIENT_EVIDENCE` |
| 3 · shards | fewer logprob shards than `world_size` | `INSUFFICIENT_EVIDENCE` — a missing shard is wrong in a way that does not show |
| 4 · guards | a pitfall guard failed | `INSUFFICIENT_EVIDENCE` |

`SwitchStatus.FELL_BACK` is the dangerous state: the reference was requested, the
engine silently reverted to native, and "the deviation did not change" then reads
as a clean `NOT_THIS_FACTOR`. Gate 1 exists for exactly this.

## Layout

```
mismatch/
├── schema/              pure data types, frozen dataclasses and enums, no behaviour
│   ├── values.py            PolicyRole, ExecutionPath, Precision, RebindCost, RequiredSetting
│   ├── collectives.py       CollectiveContract, ReductionOrder, DeterminismLevel, rewrites
│   ├── contracts.py         OperatorContract, ComparisonRule, ComparisonIssue
│   ├── factors.py           MismatchFactor, Switch, Prerequisites, ReferenceImplementation
│   ├── variants.py          FactorVariant, SwitchStatus, Diagnosis, NoiseFloor
│   ├── thresholds.py        EXPECTED_RANGES — thresholds as code, not configuration
│   ├── rollout_context.py   ComparisonIdentity, RolloutGroup, BatchPlacement
│   ├── metrics.py           MismatchMetrics, VariantResult, FactorReport, LogprobShard
│   ├── fingerprints.py      ReuseKey, ExecutionFingerprint, VariantRecord
│   ├── pitfalls.py          KnownPitfall, FailureMode
│   └── tracing.py           ModuleCorrespondence, PropagationEdge, RootCauseHypothesis
├── pipeline/            the seven steps, free functions only, no state
├── engines/             the two sides under test — megatron.py and vllm.py, nothing else
├── reference_adapters/  delivering pinned settings by channel and reading them back
├── model_meta/          per-model module correspondence and call chain (qwen3.py)
├── operator_checks/     plugins, one directory per operator — **empty by design**
└── __main__.py          CLI, and the only module that imports operator plugins
```

### What belongs in `engines/`, and what does not

`engines/` holds **exactly two modules: `megatron.py` (training side) and
`vllm.py` (rollout side)** — the two policies as they really run. Each one owns
how its engine is constructed, how a switch is delivered to it, and how the
*effective* value is read back off it. They are shared across operators: all of
attention's factors use the one `vllm.py`, and **adding an operator never adds a
file here**. Both are placeholders today; their docstrings list the settings that
must be pinned and the readback path for each.

Anything that merely satisfies the `ScoringBackend` protocol is **not** an
engine. Test harnesses live under `tests/`:

| module | what it is |
|---|---|
| `tests/mismatch_cpu_backend.py` | CPU stub for exercising the gates and the matrix with no GPU, with an injectable one-sided bias |

The line is role, not protocol: an engine is *a side under test*; a harness is
what lets you run the framework when that side is not available.

Three dependency rules hold the plugin seam open:

1. `schema/` never imports `pipeline/`; inside `schema/` the imports go one way,
   with `values.py` at the top importing nothing from the project.
2. `pipeline/` never imports `operator_checks/` — it only sees what the registry
   hands it. Break this and "add an operator" becomes "change the framework".
3. Only `__main__` imports `operator_checks/`, which is what triggers plugin
   self-registration.

## Running it

```bash
python -m rl_engine.mismatch list                    # registered operators and their factors
python -m rl_engine.mismatch plan --gpu-count 2      # expand into cases, cheapest rebuild first
python -m rl_engine.mismatch plan --json             # same, machine readable
```

`list` currently prints "no operator plugins registered", and that is the
intended state: **the framework ships without operators.** Each operator is
claimed and written separately, and adding one changes nothing outside its own
directory plus one line in `__main__._OPERATOR_PACKAGES`.

The plumbing is testable without a GPU: `tests/mismatch_cpu_backend.py` is a
scoring backend that can simulate a one-sided bias, a switch that silently does
nothing, and an implementation that is unstable across environments — so the
gates and the matrix are exercised for real in `tests/test_mismatch_framework.py`.

## Noise floors: run them in order

A factor's result only means something at a floor that can resolve it. The four
floors are arranged so **each step down adds exactly one new noise source**, which
is what makes a failure at one floor point at a known suspect set.

| floor | configuration | new noise source |
|---|---|---|
| `SINGLE_LAYER_ANCHOR` | 1 layer, single device, determinism on, one token | none — failing bitwise here is an **operator bug**, not mismatch |
| `FULL_MODEL_SINGLE_GPU` | all layers, single device | accumulation over depth |
| `SHARDED_SINGLE_NODE` | TP + SP on one node | reduction-order differences — the first floor with *real* mismatch |
| `PRODUCTION` | target TP/CP/PP, determinism off, decode path | everything else; the only floor whose numbers may be read against `EXPECTED_RANGES` |

A floor that has not passed blocks the next one.

## Adding to this package

| you want to | read |
|---|---|
| add a kernel's mismatch factor (new operator, or a factor on an existing one) | [`docs/add-a-kernel-factor.md`](docs/add-a-kernel-factor.md) |
| add a communication feature (a collective, a reduction order, a rewrite) | [`docs/add-a-comm-feature.md`](docs/add-a-comm-feature.md) |

Both are the same shape of work: **adding an operator is adding a directory,
adding a factor is adding a file**, and no existing file changes. If you find
yourself editing the planner, a global dict, or another operator's file, the
abstraction is missing something — say so and fix the framework rather than
patching in place.
