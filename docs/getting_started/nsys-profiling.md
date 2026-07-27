# NVTX & Nsight Profiling Guide

This guide explains how to see RL-Kernel's compiled C++/CUDA operators as distinct, named
blocks in an NVIDIA Nsight Systems (`nsys`) timeline. It is a profiling checklist, not a static
report — regenerate the trace on your target machine when you need fresh data, and avoid
committing generated `.nsys-rep` files to the repository.

## Scope

Every operator bound in `csrc/ops.cpp` (`fused_logp`, `deterministic_logp`,
`prefix_shared_attention`, the SM90 TMA/WGMMA linear-logp family, and their `*_out`/`*_fp32`/
`*_indexed`/`*_online` variants) is wrapped in an NVTX range named `rl_kernel::<op>` via
`csrc/utils/nvtx_utils.h`. This lets `nsys` draw one labeled block per RL-Kernel op call,
grouped above the raw CUDA kernel launches that op triggers internally.

This is a micro-level, kernel-boundary trace. For macro-level throughput, fallback-rate, and
cache-fragmentation metrics across a training/rollout cluster, see the
[Metrics & Dashboards Guide](metrics-and-dashboards.md) instead.

## 1. Build the Extension

NVTX ranges link against `libnvToolsExt` (via `-lnvToolsExt`, already wired into `setup.py`'s
CUDA build), which ships with every CUDA toolkit -- no extra install step is required beyond a
normal build:

```bash
MAX_JOBS=2 python setup.py build_ext --inplace
```

## 2. Record a Trace

Wrap any RL-Kernel entry point with `nsys profile`. The GRPO single-GPU example is a convenient
smoke workload:

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_index> nsys profile -o rlkernel_report \
  python examples/grpo_single_gpu.py \
  --device cuda \
  --require-fused-logp \
  --steps 2 \
  --num-prompts 1 \
  --samples-per-prompt 2 \
  --prompt-len 2 \
  --completion-len 3 \
  --vocab-size 16 \
  --hidden-dim 8
```

This produces `rlkernel_report.nsys-rep` in the current directory.

## 3. Inspect the Timeline

Open the report in the Nsight Systems UI (`nsys-ui rlkernel_report.nsys-rep`), or summarize it
on the command line:

```bash
nsys stats --report nvtx_sum rlkernel_report.nsys-rep
```

Confirm that named ranges appear as distinct, non-overlapping blocks on the **NVTX** row,
directly above the correlated CUDA HW kernel rows. Expect names matching the op(s) actually
exercised by the workload, for example:

- `rl_kernel::fused_logp`
- `rl_kernel::deterministic_logp`
- `rl_kernel::prefix_shared_attention`
- `rl_kernel::fused_logp_sm90` / `rl_kernel::fused_linear_logp_sm90` (Hopper-only, requires a
  build with `KERNEL_ALIGN_FORCE_SM90=1`)

Each block's duration should track the wall-clock time of that op's C++ entry point, including
every CUDA kernel it launches internally — this is what distinguishes an "op-level" NVTX block
from the finer-grained individual kernel-launch rows `nsys` already draws on its own.

## 4. Manual Verification Only

Confirming that labeled blocks render correctly in Nsight Systems is a **manual verification
step**. There is no GPU + `nsys` available in standard CI, and no meaningful unit test can
assert "nsys drew a labeled block" — so this guide, not an automated test, is the source of
truth for validating NVTX coverage. Treat a successful walkthrough of this checklist as the
acceptance bar, not a green CI run.

## Reporting Guidance

When sharing a trace or a screenshot from Nsight Systems, include:

- GPU model and compute capability.
- Driver, CUDA runtime, and `nsys` version (`nsys --version`).
- The exact `CUDA_VISIBLE_DEVICES` mapping and command used to record the trace.
- Which NVTX-named ranges were visible and whether their nesting/order matched expectations.

Keep committed docs focused on the process and commands. Generated `.nsys-rep` files and
point-in-time screenshots should stay outside the repository.
