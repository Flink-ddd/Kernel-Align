#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark Qwen3 TP-local SwiGLU forward backends on Hopper."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.cuda.activation.swiglu import SwiGLUSM90Op
from rl_engine.kernels.ops.triton.activation.swiglu import TritonSwiGLUOp


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096, help="M_local token rows")
    parser.add_argument(
        "--width", type=int, default=6144, help="Qwen3-8B TP=2 local intermediate width"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_swiglu.py requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(f"benchmark_swiglu.py requires Hopper SM90, got sm_{major}{minor}")

    generator = torch.Generator(device="cuda").manual_seed(239)
    shape = (args.rows, args.width)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    cuda_op = SwiGLUSM90Op()
    triton_op = TritonSwiGLUOp()

    timings = {
        "PyTorch": _bench(
            lambda: F.silu(gate.float()).mul(up.float()).bfloat16(),
            args.warmup,
            args.iterations,
        ),
        "CUDA SM90": _bench(lambda: cuda_op(gate, up), args.warmup, args.iterations),
        "Triton": _bench(lambda: triton_op(gate, up), args.warmup, args.iterations),
    }

    print(f"device={torch.cuda.get_device_name()} shape={shape} dtype=bf16")
    print("backend        latency_ms")
    for name, latency in timings.items():
        print(f"{name:<14} {latency:>10.4f}")


if __name__ == "__main__":
    main()
