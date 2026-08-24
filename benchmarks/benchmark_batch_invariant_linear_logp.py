# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark single-card batch-invariant fused linear log-probability.

The benchmark compares three SM90 paths:

1. the new batch-invariant fused forward;
2. the existing throughput-oriented fused ``linear_logp`` forward;
3. a batch-invariant materialized composition (LM head, then logp).

The two fused timings call their compiled symbols directly with the same prepared
inputs, so neither side includes Python validation or a host synchronization.
FP32 inputs for the materialized composition are allocated only after the fused
measurements and outside its timed region. Memory columns report incremental
operator allocations above those prepared inputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial

import torch
from tabulate import tabulate

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.linear.lm_head import SM90LMHeadOp
from rl_engine.kernels.ops.cuda.loss.batch_invariant_logp import BatchInvariantLogpSM90Op

DEFAULT_CONFIGS = [
    (1, 4096, 151936),
    (32, 4096, 151936),
    (256, 4096, 151936),
    (4096, 2048, 32768),
    (4096, 4096, 151936),
]
_HIDDEN_TILE = 32
_FP32_LOGP_VOCAB_ALIGNMENT = 4


def _make_inputs(num_tokens: int, hidden_dim: int, vocab_size: int):
    hidden = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_dim, device="cuda", dtype=torch.bfloat16)
    target_ids = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
    return hidden, weight, target_ids


def _time_ms(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _peak_memory_mb(fn: Callable[[], torch.Tensor], warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    return peak / (1024**2)


def _validate_run_counts(warmup: int, iterations: int) -> None:
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")


def _validate_config(config: tuple[int, ...]) -> tuple[int, int, int]:
    if len(config) != 3:
        raise ValueError(f"each config must contain N,D,V, got {config}")

    num_tokens, hidden_dim, vocab_size = config
    if num_tokens <= 0 or hidden_dim <= 0 or vocab_size <= 0:
        raise ValueError(f"N, D, and V must be positive, got {config}")
    if hidden_dim % _HIDDEN_TILE != 0:
        raise ValueError(f"D must be divisible by {_HIDDEN_TILE}, got {hidden_dim}")
    if vocab_size % _FP32_LOGP_VOCAB_ALIGNMENT != 0:
        raise ValueError(
            "V must be divisible by 4 so the materialized FP32 logp comparator "
            f"stays on its SM90 TMA backend, got {vocab_size}"
        )
    return num_tokens, hidden_dim, vocab_size


def _parse_configs(value: str | None) -> list[tuple[int, int, int]]:
    if value is None:
        return list(DEFAULT_CONFIGS)
    try:
        configs = [tuple(int(item) for item in group.split(",")) for group in value.split(";")]
    except ValueError as error:
        raise ValueError(
            "--configs must contain semicolon-separated integer N,D,V triples"
        ) from error
    return [_validate_config(config) for config in configs]


def _run_materialized(lm_head, logp, hidden, weight, target_ids):
    logits = lm_head(hidden, weight)
    return logp(logits, target_ids)


def _run_fused(symbol, hidden, weight, target_ids):
    logp, _lse = symbol(hidden, weight, target_ids, None)
    return logp


def run(args: argparse.Namespace) -> None:
    _validate_run_counts(args.warmup, args.iterations)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("benchmark requires an SM90 Hopper GPU")
    required_symbols = (
        "batch_invariant_linear_logp_sm90",
        "fused_linear_logp_sm90",
        "lm_head_sm90_forward",
        "lm_head_sm90_forward_fp32",
        "batch_invariant_logp_sm90",
    )
    if not _EXT_AVAILABLE or any(not hasattr(_C, name) for name in required_symbols):
        raise RuntimeError("benchmark requires all compiled SM90 comparison symbols")

    invariant_lm_head = SM90LMHeadOp()
    invariant_logp = BatchInvariantLogpSM90Op()
    rows = []

    with torch.inference_mode():
        for config in args.configs:
            num_tokens, hidden_dim, vocab_size = _validate_config(tuple(config))
            hidden, weight, target_ids = _make_inputs(num_tokens, hidden_dim, vocab_size)
            target_ids_i32 = target_ids.to(torch.int32)

            run_invariant_fused = partial(
                _run_fused,
                _C.batch_invariant_linear_logp_sm90,
                hidden,
                weight,
                target_ids_i32,
            )
            run_throughput_fused = partial(
                _run_fused,
                _C.fused_linear_logp_sm90,
                hidden,
                weight,
                target_ids_i32,
            )
            invariant_ms = _time_ms(run_invariant_fused, args.warmup, args.iterations)
            throughput_ms = _time_ms(run_throughput_fused, args.warmup, args.iterations)
            invariant_memory = _peak_memory_mb(run_invariant_fused)
            throughput_memory = _peak_memory_mb(run_throughput_fused)
            invariant_output = run_invariant_fused()
            throughput_output = run_throughput_fused()

            hidden_fp32 = hidden.float()
            weight_fp32 = weight.float()
            del (
                run_invariant_fused,
                run_throughput_fused,
                hidden,
                weight,
                target_ids_i32,
            )

            run_materialized = partial(
                _run_materialized,
                invariant_lm_head.forward_fp32,
                invariant_logp,
                hidden_fp32,
                weight_fp32,
                target_ids,
            )
            materialized_ms = _time_ms(run_materialized, args.warmup, args.iterations)
            materialized_memory = _peak_memory_mb(run_materialized)
            materialized_output = run_materialized()
            invariant_max_abs = float((invariant_output - materialized_output).abs().max())
            throughput_max_abs = float((throughput_output - materialized_output).abs().max())

            rows.append(
                [
                    f"{num_tokens}x{hidden_dim}x{vocab_size}",
                    f"{invariant_ms:.3f}",
                    f"{throughput_ms:.3f}",
                    f"{invariant_ms / throughput_ms:.2f}x",
                    f"{materialized_ms:.3f}",
                    f"{invariant_memory:.0f}",
                    f"{throughput_memory:.0f}",
                    f"{materialized_memory:.0f}",
                    f"{invariant_max_abs:.3e}",
                    f"{throughput_max_abs:.3e}",
                ]
            )
            del (
                run_materialized,
                hidden_fp32,
                weight_fp32,
                target_ids,
                invariant_output,
                throughput_output,
                materialized_output,
            )
            torch.cuda.empty_cache()

    print(
        tabulate(
            rows,
            headers=[
                "shape (N x D x V)",
                "invariant fused ms",
                "throughput fused ms",
                "invariance overhead",
                "materialized ms",
                "invariant incremental MB",
                "throughput incremental MB",
                "materialized incremental MB",
                "invariant max abs vs materialized",
                "throughput max abs vs materialized",
            ],
            tablefmt="github",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help=(
            "Semicolon-separated N,D,V triples with D divisible by 32 and V divisible by 4, "
            "for example '4096,4096,151936'"
        ),
    )
    args = parser.parse_args()
    try:
        _validate_run_counts(args.warmup, args.iterations)
        args.configs = _parse_configs(args.configs)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    run(parse_args())
