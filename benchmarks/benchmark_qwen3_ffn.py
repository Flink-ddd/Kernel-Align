# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare Qwen3 FFN consistent and fast paths on SM90.

The default shape is the TP=2 rank-local Qwen3-8B FFN from the #239
architecture. ``--intermediate-size 12288`` benchmarks the unsharded FFN;
communication is intentionally outside this PR2 benchmark.

Examples:
    python benchmarks/benchmark_qwen3_ffn.py
    python benchmarks/benchmark_qwen3_ffn.py --backend triton --profile-stages
    python benchmarks/benchmark_qwen3_ffn.py --intermediate-size 12288
    python benchmarks/benchmark_qwen3_ffn.py --tokens 512
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from tabulate import tabulate

from rl_engine.kernels.ffn import (
    QWEN3_8B_HIDDEN_SIZE,
    QWEN3_8B_INTERMEDIATE_SIZE,
    QWEN3_8B_TP2_INTERMEDIATE_SIZE,
    build_qwen3_ffn,
    qwen3_ffn_fp32_reference,
)
from rl_engine.testing.reference_ops import summarize_kernel_drift


@dataclass(frozen=True)
class FFNBenchmarkResult:
    path: str
    gemm_backend: str
    activation_backend: str
    forward_ms: float
    forward_backward_ms: float
    max_abs_error: float
    mean_abs_error: float
    peak_memory_mb: float
    stage_ms: dict[str, float]


def _time_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
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
    return float(start.elapsed_time(end)) / iters


def _make_case(args, path: str):
    device, dtype = torch.device("cuda"), torch.bfloat16
    h, i = args.hidden_size, args.intermediate_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    def make(shape, scale):
        value = torch.randn(shape, generator=generator, dtype=torch.float32)
        value.mul_(scale)
        return value.to(device=device, dtype=dtype)

    gate_weight = make((h, i), args.weight_std)
    up_weight = make((h, i), args.weight_std)
    down_weight = make((i, h), args.weight_std)
    x = make((args.tokens, h), args.input_std)
    dy = make((args.tokens, h), 1.0)
    backend = args.backend if path == "consistent" else "pytorch"
    module = build_qwen3_ffn(
        gate_weight,
        up_weight,
        down_weight,
        path=path,
        backend=backend,
    )
    return module, x, dy


def _stage_times(module, x, *, warmup: int, iters: int) -> dict[str, float]:
    with torch.no_grad():
        gate = module.gemm_op(x, module.gate_weight)
        up = module.gemm_op(x, module.up_weight)
        hidden = module.swiglu_op(gate, up)
        closures = {
            "gate_gemm": lambda: module.gemm_op(x, module.gate_weight),
            "up_gemm": lambda: module.gemm_op(x, module.up_weight),
            "swiglu": lambda: module.swiglu_op(gate, up),
            "down_gemm": lambda: module.gemm_op(hidden, module.down_weight),
        }
        return {
            name: _time_ms(closure, warmup=warmup, iters=iters)
            for name, closure in closures.items()
        }


def _run_path(args, path: str) -> FFNBenchmarkResult:
    module, x, dy = _make_case(args, path)

    with torch.no_grad():
        candidate = module(x)
        reference = qwen3_ffn_fp32_reference(
            x, module.gate_weight, module.up_weight, module.down_weight
        ).output
        drift = summarize_kernel_drift(candidate, reference)
    del candidate, reference

    def forward():
        with torch.no_grad():
            return module(x)

    def forward_backward():
        module.zero_grad(set_to_none=True)
        x.grad = None
        module(x).backward(dy)
        return x.grad

    x.requires_grad_(True)
    forward_ms = _time_ms(forward, warmup=args.warmup, iters=args.iters)
    forward_backward_ms = _time_ms(forward_backward, warmup=args.warmup, iters=args.iters)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    forward_backward()
    torch.cuda.synchronize()
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    stage_ms = (
        _stage_times(module, x.detach(), warmup=args.warmup, iters=args.iters)
        if args.profile_stages
        else {}
    )
    provenance = module.provenance
    return FFNBenchmarkResult(
        path=provenance.path,
        gemm_backend=provenance.gemm_backend,
        activation_backend=provenance.activation_backend,
        forward_ms=forward_ms,
        forward_backward_ms=forward_backward_ms,
        max_abs_error=float(drift["max_abs_error"]),
        mean_abs_error=float(drift["mean_abs_error"]),
        peak_memory_mb=float(peak_memory_mb),
        stage_ms=stage_ms,
    )


def _metadata(args) -> dict[str, object]:
    capability = torch.cuda.get_device_capability()
    return {
        "model": "Qwen3-8B dense FFN",
        "device": torch.cuda.get_device_name(),
        "compute_capability": f"SM{capability[0]}{capability[1]}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "intermediate_scope": (
            "tp2_local"
            if args.intermediate_size == QWEN3_8B_TP2_INTERMEDIATE_SIZE
            else "custom_or_unsharded"
        ),
        "seed": args.seed,
        "data_generator": "CPU FP32 MT19937, then quantized to CUDA BF16",
        "input_std": args.input_std,
        "weight_std": args.weight_std,
        "warmup": args.warmup,
        "iters": args.iters,
        "scope": "rank-local FFN arithmetic; no collective communication",
    }


def render_results(results: list[FFNBenchmarkResult]) -> str:
    fast = next(result for result in results if result.path == "fast")
    fast_backward_ms = fast.forward_backward_ms
    rows = []
    for result in results:
        backward_overhead = result.forward_backward_ms / fast_backward_ms
        rows.append(
            [
                result.path,
                result.gemm_backend,
                result.activation_backend,
                f"{result.forward_ms:.3f}",
                f"{result.forward_backward_ms:.3f}",
                f"{result.forward_ms / fast.forward_ms:.2f}x",
                f"{backward_overhead:.2f}x",
                f"{result.max_abs_error:.6g}",
                f"{result.peak_memory_mb:.0f}",
            ]
        )
    return tabulate(
        rows,
        headers=(
            "path",
            "GEMM",
            "activation",
            "forward ms",
            "fwd+bwd ms",
            "forward / fast",
            "fwd+bwd / fast",
            "max abs vs FP32",
            "peak MB",
        ),
        tablefmt="github",
    )


def run(args) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 FFN benchmark requires a CUDA SM90 GPU")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 9:
        actual_sm = f"SM{capability[0]}{capability[1]}"
        raise RuntimeError(f"FFN benchmark targets SM90, got {actual_sm}")
    dimensions = (args.tokens, args.hidden_size, args.intermediate_size)
    if any(value <= 0 for value in dimensions):
        raise ValueError("tokens and FFN dimensions must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative; iters must be positive")

    torch.backends.cuda.matmul.allow_tf32 = False
    results = [_run_path(args, "consistent"), _run_path(args, "fast")]
    payload = {
        "metadata": _metadata(args),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload["metadata"], indent=2))
    print()
    print(render_results(results))
    if args.profile_stages:
        print("\nStage breakdown (ms)")
        stage_rows = [
            [
                result.path,
                *[f"{result.stage_ms[name]:.3f}" for name in result.stage_ms],
            ]
            for result in results
        ]
        stage_names = list(results[0].stage_ms)
        headers = ("path", *stage_names)
        print(tabulate(stage_rows, headers=headers, tablefmt="github"))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, indent=2) + "\n"
        args.json_out.write_text(content, encoding="utf-8")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    backends = ("cuda", "triton")
    parser.add_argument("--backend", choices=backends, default="cuda")
    parser.add_argument("--tokens", type=int, default=128)
    hidden_default = QWEN3_8B_HIDDEN_SIZE
    parser.add_argument("--hidden-size", type=int, default=hidden_default)
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=QWEN3_8B_TP2_INTERMEDIATE_SIZE,
        help=(
            "rank-local intermediate width; defaults to Qwen3-8B TP=2 "
            f"({QWEN3_8B_TP2_INTERMEDIATE_SIZE}); use "
            f"{QWEN3_8B_INTERMEDIATE_SIZE} for the unsharded FFN"
        ),
    )
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--input-std", type=float, default=1.0)
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
