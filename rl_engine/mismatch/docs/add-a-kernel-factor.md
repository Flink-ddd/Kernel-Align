<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 RL-Kernel Contributors -->

# Add a kernel's mismatch factor

You suspect a kernel of computing something different on the two sides and want
that measured and attributed instead of argued about. Worked example:
`attn.rope_fusion`.

Read [`../README.md`](../README.md) first — the four arms, the four gates and the
noise floors are assumed here. You will write **declarations only**; the
framework expands the arms, runs them, gates them, and concludes.

## Three decisions before writing anything

**Sweep or swap?** `reference is None` makes it a sweep (one arm per allowed
value); a `ReferenceImplementation` makes it a swap (the four arms). There is no
`kind` field — derivable state is state that can disagree with itself. RoPE is a
swap; `operator_checks/logprob/` has the worked sweep.

**Which reference?** `ReferenceAuthority` is a decision order:

```
FP64_ORACLE      slow, exact — gold standard at the lowest floor
     ↑ validates
SHARED_BACKEND   TransformerEngine / FlashInfer — look here FIRST
     ↑ falls back to
SELF_WRITTEN     only when the first two have a real semantic hole
```

A PR adding a `SELF_WRITTEN` reference must say why the first two cannot cover
it. RoPE is covered by TE and FlashInfer, so it is `SHARED_BACKEND`.

**Which floor?** The lowest one where the factor is not identically zero. A
`PROCESS_GROUP_REBUILD` switch is identical on a single device, so running it at
the anchor floor only burns machine time.

## Step 1 — the directory

Skip to step 3 if the operator exists; then you are only adding one file.

```
operator_checks/attention/
├── __init__.py     operator name + discover_factors
├── adapter.py      the four operator-level methods
├── _common.py      reference implementations, contract helpers
└── factors/
    ├── __init__.py
    └── rope_fusion.py
```

**A factor file's name is its id with the operator prefix stripped.**
`discover_factors()` enforces it, so renaming an id without renaming the file
fails at import rather than silently dropping the factor.

## Step 2 — `_common.py`

```python
TE_ROPE_REFERENCE = ReferenceImplementation(
    name="transformer_engine",
    tier=ReferenceAuthority.SHARED_BACKEND,
    training_impl="transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb",
    rollout_impl="flashinfer.rope.apply_rope",
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
    ),
    required_settings=(
        RequiredSetting(
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0",
            SettingChannel.ENV_VAR, readback="os.environ",
        ),
    ),
    pinned_libraries=(LibraryPin("transformer_engine", "2.9.0.dev0", commit="8260f49"),),
)
```

Three fields decide more than they look like they do:

- **`covers_paths` defines the shape of the self-check gate**: every path this
  reference covers must agree bitwise on the same sequence. Two paths put the
  gate across the two sides; a reference that also covers `ROLLOUT_DECODE` puts
  it *inside* the rollout side, and then no decode stub is needed on the training
  side.
- **`required_settings` are not documentation.** Each is delivered by its channel
  and verified by readback. `readback=None` can only ever be `UNOBSERVABLE`.
- **`pinned_libraries` is required.** TE and FlashInfer change kernel selection
  across versions, so the same factor can reach the opposite conclusion on a
  different one.

## Step 3 — the factor file

One `FACTOR` constant, no behaviour. See
`operator_checks/attention/factors/rope_fusion.py` for the full declaration; the
fields worth thinking about:

**`comparison_rules`** — keys are dotted paths from the contract root
(`precision.accumulate`, `collectives[0].reduction_order`, `extra.rope_theta`).
The framework indexes both contracts by path, so you never write a "collect the
comparable fields" function.

| tier | for | consequence |
|---|---|---|
| `MUST_MATCH_BITWISE` | identity: shapes, dtypes, TP size | differing ⇒ the case is void, not a finding |
| `MUST_MATCH_SEMANTICALLY` | may be implemented differently, must mean the same | differing ⇒ a `SEMANTIC_MISMATCH` |
| `RECORD_ONLY` | representation that differs by construction | recorded, never compared |

`RECORD_ONLY` exists so structural differences do not drown the real problems.
Declare a packed-QKV-style field `MUST_MATCH_*` and your factor disappears under
false positives. The registry also rejects two factors declaring the same
contract field at different tiers — if that fires, one declaration is wrong.

**`required_evidence`** — three items apply to every factor; operator-specific
ones are plain string constants (`POSITION_CACHE`, `LSE_EXPORT`, …) so adding an
operator never means editing an enum. Missing evidence is gate 2, which is *not*
the same verdict as "measured and clean".

**`prerequisites`** — a declared whitelist. The planner turns each unmet item
into a reason, so `plan` prints *why* a factor was skipped.

**`pitfalls`** — `symptom` and `actual_cause` are separate because a pitfall is
one precisely when its appearance points at the wrong cause. `guard_runs_at`
should be the lowest floor that can run the check.

## Step 4 — `adapter.py`

The four methods are operator-level, not factor-level: reading configuration back
from an engine is the same logic for all of an operator's factors. See
`operator_checks/attention/adapter.py` for the signatures.

`resolve_implementation` is where a factor most often dies quietly. Returning a
bare `None` produces `FELL_BACK` with nothing to investigate, and a fallen-back
arm whose deviation "did not change" reads exactly like a clean
`NOT_THIS_FACTOR`. **Return the trace even when resolution fails.**

## Step 5 — register

```python
@OPERATOR_CHECKS.register
class AttentionChecks:
    operator = "attention"

    def declare_factors(self):
        return discover_factors(__package__)

    build_contract = staticmethod(adapter.build_contract)
    ...
```

Then add the package to `__main__._OPERATOR_PACKAGES` — the one edit outside your
own directory. Adding the next factor drops a file into `factors/` and changes
neither file.

## Step 6 — check it, without a GPU

```bash
python -m rl_engine.mismatch list
# attention: 1 factors
#   attn.rope_fusion                         kernel_implementation
```

`plan` on a machine without the prerequisites names every unmet one rather than
silently omitting the factor:

```
skipped (prerequisites not met):
  attn.rope_fusion: operator 'rope' is not dispatchable
  attn.rope_fusion: package 'transformer_engine>=2.0' is not installed
```

Where they are met, the same declaration expands into the four arms ordered
cheapest-rebuild-first. Then add a test: `tests/mismatch_cpu_backend.py` can
inject a one-sided bias, silently ignore a switch, and be unstable across
environments — enough to prove your factor attributes the right side and that a
fallback is reported rather than mistaken for a clean result.

## What you do not have to write

`build_variants()`, `compare_contracts()`, `diagnose()`, `missing_prerequisites()`,
`order_cases_by_rebind_cost()`, and `repeat_under` + `assert_order_is_topology_independent()`
for topology-invariance reruns.

## Checklist

- [ ] File name equals the factor id minus the operator prefix.
- [ ] `question` is one line and says what the factor *answers*.
- [ ] Representation-only fields are `RECORD_ONLY`, not `MUST_MATCH_*`.
- [ ] A `SELF_WRITTEN` reference is justified in the PR body.
- [ ] Every `RequiredSetting` has a `readback`, or `UNOBSERVABLE` is accepted.
- [ ] `pinned_libraries` names an exact version.
- [ ] Pitfalls are `KnownPitfall` values, not comments.
- [ ] `list` and `plan` show what you expect.
