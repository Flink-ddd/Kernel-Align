<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 RL-Kernel Contributors -->

# `rl_engine/mismatch` tutorials

Start with the package overview in [`../README.md`](../README.md) — what a factor
is, the four arms, the four gates, and the seven pipeline steps. These tutorials
assume it.

| tutorial | when you need it |
|---|---|
| [add-a-kernel-factor.md](add-a-kernel-factor.md) | a kernel (attention, GEMM, RoPE, logprob, SwiGLU, …) is suspected of computing something different on the two sides, and you want it measured and attributed |
| [add-a-comm-feature.md](add-a-comm-feature.md) | the suspect is a **collective**: a reduction order, an `all_reduce` → `reduce_scatter + all_gather` rewrite, a CP/split-K merge, a communication backend |

Both end with the same two commands, which are how you check your work without a
GPU:

```bash
python -m rl_engine.mismatch list
python -m rl_engine.mismatch plan --gpu-count 2
```

## The rule both tutorials follow

**Adding an operator is adding a directory. Adding a factor is adding a file.**
No existing file changes, apart from one line in `__main__._OPERATOR_PACKAGES`
when the operator is new.

If your change needs an edit inside `pipeline/`, a global dict, or another
operator's directory, stop: the framework is missing an abstraction. Raise it and
fix the framework, rather than patching around it — the seam is the only thing
keeping forty-odd factors from turning into forty-odd special cases.
