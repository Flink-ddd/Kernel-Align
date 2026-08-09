<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 RL-Kernel Contributors -->

# Tutorial: add a kernel's mismatch factor

You suspect a kernel — attention, GEMM, RoPE, the logprob path, SwiGLU — of
computing something different on the training side than on the rollout side, and
you want that suspicion **measured and attributed to a side** instead of argued
about.

This tutorial walks the whole path with one worked example: `attn.rope_fusion`,
"is the difference caused by fused RoPE versus small-operator RoPE, or by
`position_ids` / `theta` / the cast boundary?"

By the end you will have written **declarations only** — no execution logic. The
framework expands the arms, runs them, gates them, and draws the conclusion.

## Before you write anything: three decisions

**1 · Is this a parameter sweep or an implementation swap?**

| | you declare | the framework expands into |
|---|---|---|
| **sweep** — nothing is replaced, you scan a setting | `Switch.allowed_values`, `reference=None` | one arm per allowed value |
| **swap** — a reference implementation replaces the native one | `Switch` + `ReferenceImplementation` | the four arms (plus an `fp64_oracle` arm if declared) |

There is deliberately no `kind` field: `reference is None` *is* the distinction,
because derivable state is state that can disagree with itself. RoPE is a swap;
`logp.precision_downcast` (in `operator_checks/logprob/`) is the worked sweep.

**Know what a sweep gives up.** `diagnose()` runs the four-arm matrix, which
needs a `both_native` baseline and the two one-sided arms. A sweep has neither,
so it returns `INSUFFICIENT_EVIDENCE` — *by design*: with nothing swapped there is
no side to attribute to. A sweep measures and records; it does not conclude. Read
its numbers against `EXPECTED_RANGES` yourself, or declare explicit `variants`
once a reference implementation exists.

**2 · Which reference implementation, and are you allowed to write one?**

`ReferenceAuthority` is a decision order, not a description:

```
FP64_ORACLE      slow, mathematically exact — gold standard at the lowest floor
     ↑ validates
SHARED_BACKEND   TransformerEngine (training) / FlashInfer (rollout) — look here FIRST
     ↑ falls back to
SELF_WRITTEN     only when the first two have a real semantic hole
```

**A PR adding a `SELF_WRITTEN` reference must say why the first two cannot cover
it.** RoPE is covered by TE (`pytorch/attention/rope.py`) and by
`flashinfer.rope`, so it is `SHARED_BACKEND`.

**3 · Which noise floor can even show it?**

Pick the *lowest* floor at which the factor is not identically zero. RoPE shows up
with one layer on one device, so `SINGLE_LAYER_ANCHOR` — cheap, and a bitwise
failure there is an operator bug rather than mismatch. A factor whose switch is
`PROCESS_GROUP_REBUILD` is identical on a single device; running it at the anchor
floor only burns machine time (`planner.suggested_floor_is_lowest`).

## Step 1 · Create the operator directory

Skip to step 3 if the operator already exists — then you are only adding one file.

```
rl_engine/mismatch/operator_checks/attention/
├── __init__.py          ~15 lines: operator name + discover_factors, everything else delegated
├── adapter.py           the four operator-level methods
├── _common.py           shared across factors: reference implementations, contract helpers
└── factors/
    ├── __init__.py      empty
    └── rope_fusion.py   one FACTOR constant, 30–50 lines
```

Why one file per factor: an operator has a dozen-plus factors of eight-odd fields
each. In one file that is 800 lines nobody wants to edit.

**The file name is the factor id with the operator prefix stripped.**
`discover_factors()` enforces it, so renaming an id without renaming the file
fails at import instead of silently dropping the factor.

## Step 2 · `_common.py` — what the whole operator shares

```python
# operator_checks/attention/_common.py
from rl_engine.mismatch.schema import (
    ExecutionPath,
    LibraryPin,
    ReferenceAuthority,
    ReferenceImplementation,
    RequiredSetting,
    SettingChannel,
)

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
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            "0",
            SettingChannel.ENV_VAR,
            readback="os.environ",
        ),
    ),
    pinned_libraries=(
        LibraryPin("transformer_engine", "2.3.0", container_digest="sha256:..."),
    ),
)
```

Three fields decide more than they look like they do:

- **`covers_paths` defines the shape of the self-check gate.** The gate is not
  "both sides use the same implementation", it is "*every path this reference
  covers must agree bitwise on the same sequence*". Two paths ⇒ the gate holds
  across the two sides. A reference that also covers `ROLLOUT_DECODE` (FlashInfer
  attention does) makes the gate hold *inside the rollout side*, and then you do
  not need a decode stub on the training side at all.
- **`required_settings` are not documentation.** Each is delivered by its channel
  (`reference_adapters/settings.py::apply_required_settings`) and verified by
  `verify_required_settings`. A setting with `readback=None` can only ever be
  recorded `UNOBSERVABLE` — delivered but unprovable is the same as not
  delivered.
- **`pinned_libraries` is required.** TE and FlashInfer change kernel selection
  across versions; the same factor can reach the *opposite* conclusion on a
  different version. The pin goes into the execution fingerprint, so bumping it
  invalidates historical results instead of quietly making them incomparable.

## Step 3 · The factor file — declarations, no behaviour

```python
# operator_checks/attention/factors/rope_fusion.py
from rl_engine.mismatch.operator_checks.attention._common import TE_ROPE_REFERENCE
from rl_engine.mismatch.schema import (
    POSITION_CACHE,
    ComparisonRule,
    Evidence,
    FactorCategory,
    FailureMode,
    KnownPitfall,
    MismatchFactor,
    NoiseFloor,
    PolicyRole,
    Prerequisites,
    RebindCost,
    Switch,
)

FACTOR = MismatchFactor(
    id="attn.rope_fusion",
    operator="attention",
    category=FactorCategory.KERNEL_IMPLEMENTATION,
    question=(
        "Does the deviation come from fused vs small-operator vs sin/cos-cached "
        "RoPE, or from position_ids / theta / the cast boundary?"
    ),
    switch=Switch(
        path="attn.rope_fusion",
        rebind_cost=RebindCost.ENGINE_REBUILD,
        applies_to=(PolicyRole.ROLLOUT, PolicyRole.TRAINING),
        allowed_values=("native", "transformer_engine"),
    ),
    comparison_rules={
        "extra.rope_theta": ComparisonRule.MUST_MATCH_BITWISE,
        "extra.position_ids_digest": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.post_rope_qk_digest": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "precision.downcast_at": ComparisonRule.MUST_MATCH_SEMANTICALLY,
        "extra.fusion_boundary": ComparisonRule.RECORD_ONLY,
    },
    prerequisites=Prerequisites(
        required_ops=("rope",),
        required_packages=("transformer_engine>=2.0",),
    ),
    required_evidence=(Evidence.EFFECTIVE_CONFIG_READBACK.value, POSITION_CACHE),
    reference=TE_ROPE_REFERENCE,
    pitfalls=(
        KnownPitfall(
            id="rope_hook_not_covered",
            mode=FailureMode.MISSING_INSTRUMENTATION,
            symptom="RoPE looks perfectly consistent between the two sides",
            actual_cause="the hook never attached, so nothing was captured at all",
            guard="dump post-RoPE Q/K on both sides and compare bitwise before ablating",
            guard_runs_at=NoiseFloor.SINGLE_LAYER_ANCHOR,
        ),
    ),
)
```

### Filling the fields well

**`comparison_rules` — the tier matters more than the field list.**
Keys are dotted paths from the contract root: `precision.accumulate`,
`collectives[0].reduction_order`, `extra.rope_theta`. The framework indexes both
contracts by path and compares entry by entry, so you never write a "collect the
comparable fields" function — you only put fields in the right place in
`build_contract()`.

| tier | use it when | consequence |
|---|---|---|
| `MUST_MATCH_BITWISE` | identity: shapes, dtypes, TP size, vocab size | differing ⇒ the case is void, not a finding |
| `MUST_MATCH_SEMANTICALLY` | the two sides may implement it differently but must mean the same thing | differing ⇒ a `SEMANTIC_MISMATCH` issue |
| `RECORD_ONLY` | representation that differs by construction — packed QKV layout, page tables, backend names | recorded, **never compared** |

`RECORD_ONLY` exists precisely so structural differences do not drown the real
problems. Declare a packed-QKV-style field `MUST_MATCH_*` and your factor
disappears under false positives.

One more constraint the registry enforces: **two factors may not declare the same
contract field at two different rules.** If that fires, one of the two
declarations is wrong — do not "fix" it by renaming the path.

**`required_evidence` — what must exist before a verdict is allowed.**
Three items apply to every factor (`Evidence.EFFECTIVE_CONFIG_READBACK`,
`MODEL_STATE_FINGERPRINT`, `LIBRARY_VERSIONS`). Operator-specific evidence is a
plain string constant (`POSITION_CACHE`, `LSE_EXPORT`, `VOCAB_SHARD_MAP`,
`COLLECTIVE_CONTRACT`, …) so that adding an operator never means editing an enum
in the framework. Missing evidence is gate 2: `INSUFFICIENT_EVIDENCE`, which is
**not** the same verdict as "measured and clean".

**`prerequisites` — a whitelist, declared, not probed by hand.**
`required_ops`, `min_gpu_count`, `required_packages`, `required_model_traits`
(`"moe"`, `"linear_attention"`), `blocked_by` (issues this waits on). The planner
turns each unmet item into a reason string, so `plan` prints *why* a factor was
skipped instead of silently omitting it.

**`pitfalls` — prose gets read once; data gets enforced.**
`symptom` and `actual_cause` are separate fields on purpose: a pitfall is a
pitfall because its appearance points at the wrong cause. `guard_runs_at` should
be the lowest floor that can run the check — cheap guards first.

## Step 4 · `adapter.py` — the four operator-level methods

These are operator-level, not factor-level: how to read config back from an
engine is one piece of logic shared by all of the operator's factors.

```python
# operator_checks/attention/adapter.py
from typing import Any, Callable, Mapping

from rl_engine.mismatch.schema import (
    DowncastPoint,
    ImplementationResolution,
    OperatorContract,
    PolicyRole,
    Precision,
    PrecisionProfile,
    RejectedCandidate,
)


def build_contract(role: PolicyRole, switch_values: Mapping[str, Any]) -> OperatorContract:
    """This side's switch values -> this side's numerical contract."""
    return OperatorContract(
        operator="attention",
        role=role,
        precision=PrecisionProfile(
            compute=Precision.BF16,
            accumulate=Precision.FP32,
            downcast_at=DowncastPoint.FINAL_WRITE,
            softmax_accumulate=Precision.FP32,
        ),
        collectives=(),                       # see the comm tutorial
        extra={                               # keep flat: paths stay short and readable
            "rope_theta": 1_000_000.0,
            "position_ids_digest": "...",
            "post_rope_qk_digest": "...",
            "fusion_boundary": switch_values.get("attn.rope_fusion", "native"),
        },
    )


def read_effective_config(role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
    """Read switches back **from the engine**. A requested value is not evidence."""
    ...


def observe_collectives(role: PolicyRole, adapter: Any):
    """Which collectives actually ran. RoPE has none."""
    return ()


def resolve_implementation(
    factor_id: str, role: PolicyRole, impl_name: str
) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
    """Resolve the name an arm asks for into a callable.

    Return the trace **even when resolution fails**: which candidates were tried
    and why each was rejected.
    """
    rejected: list[RejectedCandidate] = []
    for candidate in _candidates_for(impl_name):
        try:
            return _import(candidate), ImplementationResolution(impl_name, candidate)
        except ImportError as exc:
            rejected.append(RejectedCandidate(candidate, str(exc)))
    return None, ImplementationResolution(impl_name, None, tuple(rejected))
```

`resolve_implementation` is where a factor most often dies quietly. Returning a
bare `None` produces `SwitchStatus.FELL_BACK` with nothing to investigate, and a
fallen-back arm whose deviation "did not change" reads exactly like a clean
`NOT_THIS_FACTOR`. Gate 1 catches the status; only your trace explains it.

## Step 5 · `__init__.py` — register, and nothing else

```python
# operator_checks/attention/__init__.py
from rl_engine.mismatch.operator_checks.attention import adapter
from rl_engine.mismatch.pipeline import OPERATOR_CHECKS, discover_factors


@OPERATOR_CHECKS.register
class AttentionChecks:
    operator = "attention"

    def declare_factors(self):
        return discover_factors(__package__)      # scans factors/*.py

    build_contract = staticmethod(adapter.build_contract)
    read_effective_config = staticmethod(adapter.read_effective_config)
    observe_collectives = staticmethod(adapter.observe_collectives)
    resolve_implementation = staticmethod(adapter.resolve_implementation)
```

Adding the next factor drops a file into `factors/`; this file does not change.

## Step 6 · Make the operator exist

The framework does not know which operators exist. `__main__` is the only module
that imports plugins, which is what keeps `pipeline/` from ever reaching into
`operator_checks/`:

```python
# rl_engine/mismatch/__main__.py
_OPERATOR_PACKAGES: tuple[str, ...] = (
    "rl_engine.mismatch.operator_checks.attention",
)
```

An operator that is not listed here simply does not exist as far as the framework
is concerned. **This one line is the only edit outside your own directory.**

## Step 7 · Check it, without a GPU

```bash
python -m rl_engine.mismatch list
# attention: 1 factors
#   attn.rope_fusion                         kernel_implementation
```

`plan` on a workstation without TransformerEngine reports the factor as skipped,
naming every unmet prerequisite rather than silently omitting it:

```bash
python -m rl_engine.mismatch plan --gpu-count 1
# noise floor: single_layer_anchor
# runnable factors: 0   cases: 0
#
# skipped (prerequisites not met):
#   attn.rope_fusion: operator 'rope' is not dispatchable
#   attn.rope_fusion: package 'transformer_engine>=2.0' is not installed
```

That is a statement about this machine, not a defect in the factor. Where the
prerequisites are met, the same declaration expands into the four arms, ordered
cheapest-rebuild-first:

```
# runnable factors: 1   cases: 4
# cases in execution order (cheapest rebuild first):
#   [engine_rebuild        ] attn.rope_fusion :: both_native
#   [engine_rebuild        ] attn.rope_fusion :: both_reference
#   [engine_rebuild        ] attn.rope_fusion :: training_reference_only
#   [engine_rebuild        ] attn.rope_fusion :: rollout_reference_only
```

`plan --json` gives the same thing machine readably.

Then add a test beside `tests/test_mismatch_framework.py`. The CPU backend in
`tests/mismatch_cpu_backend.py` can inject a one-sided bias, silently ignore a switch,
and be unstable across environments — enough to prove your factor attributes the
right side, and that a fallback is reported rather than mistaken for a clean
result.

## What the framework does so you do not have to

| you might reach for | it already exists |
|---|---|
| writing the four arms out by hand | `build_variants()` |
| a "compare these fields" helper | `compare_contracts()`, driven by `comparison_rules` |
| deciding whether the numbers converged | `diagnose()` — four gates, then the matrix, judged on `clip_fraction` |
| checking prerequisites and printing why one was skipped | `missing_prerequisites()` |
| ordering runs so engines get reused | `order_cases_by_rebind_cost()` |
| re-running under several `NCCL_ALGO` values and asserting bitwise equality | `FactorVariant.repeat_under` + `assert_order_is_topology_independent()` |

## Checklist before the PR

- [ ] File name equals the factor id minus the operator prefix.
- [ ] `question` is one line and says what the factor *answers*.
- [ ] Every representation-only field is `RECORD_ONLY`, not `MUST_MATCH_*`.
- [ ] A `SELF_WRITTEN` reference is justified against `SHARED_BACKEND` in the PR body.
- [ ] Every `RequiredSetting` has a `readback`, or you have accepted `UNOBSERVABLE`.
- [ ] `pinned_libraries` names an exact version, ideally with a container digest.
- [ ] Known pitfalls are encoded as `KnownPitfall`, not written in a comment.
- [ ] `python -m rl_engine.mismatch list` and `plan` both show what you expect.
