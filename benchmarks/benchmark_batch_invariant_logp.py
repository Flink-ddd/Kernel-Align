# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark batch-invariant logp: Native vs Triton vs CUDA (SM90 TMA).

All three backends compute ``logits[t, target[t]] - logsumexp(logits[t, :])``
from a materialized ``[N, V]`` logits tensor with a locked, per-row reduction
order (batch-invariant). The comparison here is latency and peak VRAM across a
vocab sweep:

- Native materializes ``log_softmax`` over the full ``[N, V]`` tensor.
- Triton streams the vocab through an online softmax (grid = one program/row).
- CUDA is the Hopper TMA online-softmax kernel (one CTA/row); only present when
  the extension is built with ``KERNEL_ALIGN_FORCE_SM90=1`` on an SM90 device.

By default only the forward pass is timed; pass ``--backward`` to also emit the
forward+backward table (grad w.r.t. logits). Timing uses ``torch.cuda`` events,
so the benchmark runs on CUDA/ROCm devices.

Usage:
    python benchmarks/benchmark_batch_invariant_logp.py
    python benchmarks/benchmark_batch_invariant_logp.py --backward
    python benchmarks/benchmark_batch_invariant_logp.py --configs "4096,128256;8192,151936"
"""

import argparse

import torch
from tabulate import tabulate

from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import NativeBatchInvariantLogpOp
from rl_engine.kernels.ops.triton.loss.batch_invariant_logp import TritonBatchInvariantLogpOp
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger


def _maybe_sm90_op():
    """The Hopper TMA op, or None when unavailable (non-Hopper / not built)."""
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
        and _EXT_AVAILABLE
        and hasattr(_C, "batch_invariant_logp_sm90")
    ):
        return None
    from rl_engine.kernels.ops.cuda.loss.batch_invariant_logp import BatchInvariantLogpSM90Op

    return BatchInvariantLogpSM90Op()


# (num_tokens, vocab); vocab kept a multiple of 8 so the bf16 TMA path runs.
DEFAULT_CONFIGS = [
    (4096, 32768),
    (4096, 128256),
    (4096, 151936),
    (8192, 128256),
]


def _make_inputs(num_tokens, vocab, device, dtype):
    logits = torch.randn(num_tokens, vocab, device=device, dtype=dtype)
    target = torch.randint(0, vocab, (num_tokens,), device=device)
    return logits, target


def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_vram_gb(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / (1024**3)


def _forward_closure(op, logits, target):
    def run():
        with torch.no_grad():
            op(logits, target, validate=False)

    return run


def _forward_backward_closure(op, logits, target):
    # Fresh leaf each call so the grad is allocated (and freed) per iteration,
    # matching a real training step and giving an honest peak-VRAM reading.
    def run():
        x = logits.detach().requires_grad_(True)
        op(x, target, validate=False).sum().backward()

    return run


def _bench_table(configs, backends, closure_factory, device, dtype, warmup, iters):
    """Build a github-markdown table timing each backend, plus CUDA speedups.

    ``backends`` is ``(native_op, triton_op, sm90_op_or_None)``; ``closure_factory``
    maps ``(op, logits, target) -> callable`` (forward-only or forward+backward).
    """
    native, triton_op, sm90_op = backends
    label = "fwd" if closure_factory is _forward_closure else "fwd+bwd"
    rows = []
    for num_tokens, vocab in configs:
        logits, target = _make_inputs(num_tokens, vocab, device, dtype)
        n_c = closure_factory(native, logits, target)
        t_c = closure_factory(triton_op, logits, target)

        n_ms = _time_ms(n_c, warmup, iters)
        t_ms = _time_ms(t_c, warmup, iters)
        n_mb = _peak_vram_gb(n_c, warmup, iters) * 1024
        t_mb = _peak_vram_gb(t_c, warmup, iters) * 1024

        row = [
            f"{num_tokens}x{vocab}",
            f"{n_ms:.3f}",
            f"{t_ms:.3f}",
            f"{n_ms/t_ms:.2f}x",
            f"{n_mb:.0f}",
            f"{t_mb:.0f}",
        ]
        if sm90_op is not None:
            s_c = closure_factory(sm90_op, logits, target)
            s_ms = _time_ms(s_c, warmup, iters)
            s_mb = _peak_vram_gb(s_c, warmup, iters) * 1024
            row += [f"{s_ms:.3f}", f"{n_ms/s_ms:.2f}x", f"{t_ms/s_ms:.2f}x", f"{s_mb:.0f}"]
        rows.append(row)

    headers = [
        "shape (N x V)",
        f"native {label} ms",
        f"triton {label} ms",
        f"{label} speedup",
        f"native {label} MB",
        f"triton {label} MB",
    ]
    if sm90_op is not None:
        headers += [f"cuda {label} ms", "cuda vs native", "cuda vs triton", f"cuda {label} MB"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def run_benchmark(args):
    if device_ctx.device_type not in ["cuda", "hip"]:
        raise RuntimeError(
            "batch_invariant_logp benchmark requires a CUDA/ROCm GPU (uses torch.cuda timing)."
        )

    device = device_ctx.device
    dtype = torch.bfloat16
    backends = (NativeBatchInvariantLogpOp(), TritonBatchInvariantLogpOp(), _maybe_sm90_op())

    logger.info(
        f"batch_invariant_logp benchmark on {device} (dtype={dtype}); "
        f"SM90 TMA backend {'enabled' if backends[2] is not None else 'unavailable'}"
    )

    print("Forward")
    _bench_table(args.configs, backends, _forward_closure, device, dtype, args.warmup, args.iters)
    if args.backward:
        print("\nForward + backward")
        _bench_table(
            args.configs,
            backends,
            _forward_backward_closure,
            device,
            dtype,
            args.warmup,
            args.iters,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Also time and measure a forward+backward pass (grad w.r.t. logits).",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Semicolon-separated 'tokens,vocab' tuples, e.g. '4096,128256;8192,151936'.",
    )
    args = parser.parse_args()
    if args.configs:
        args.configs = [tuple(int(x) for x in tup.split(",")) for tup in args.configs.split(";")]
    else:
        args.configs = DEFAULT_CONFIGS
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
