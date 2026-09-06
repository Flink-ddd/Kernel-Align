# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark bypassing the unused ROCm RoPE autograd wrapper in rollout.

Both arms build the same FP32 cos/sin table and launch the same deterministic
HIP kernels. The baseline reproduces the previous ``Function.apply`` path;
the candidate calls the shared forward arithmetic directly under inference
mode. Query and key outputs must remain byte-identical.

Example on one MI300X:

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/benchmark_rocm_rope_inference_dispatch.py \
      --tokens 1,2,32 --blocks 10 --iterations 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable

import torch

from rl_engine.kernels.ops.rocm.rotary_embedding.rope import (
    RocmDeterministicRoPEOp,
    _RocmRoPEPairFunction,
)


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _elapsed_ms(function: Callable[[], object], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _benchmark_tokens(
    *,
    tokens: int,
    query_heads: int,
    key_heads: int,
    head_dim: int,
    warmup: int,
    blocks: int,
    iterations: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(390_600 + tokens)
    query = torch.randn(
        query_heads,
        tokens,
        head_dim,
        dtype=torch.bfloat16,
        generator=generator,
    ).cuda()
    key = torch.randn(
        key_heads,
        tokens,
        head_dim,
        dtype=torch.bfloat16,
        generator=generator,
    ).cuda()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    operator = RocmDeterministicRoPEOp()

    @torch.inference_mode()
    def autograd_wrapper() -> tuple[torch.Tensor, torch.Tensor]:
        # Reproduce the complete public method before the fast path, including
        # its validation and independent detach semantics.
        operator._validate_input(query)
        operator._validate_input(key)
        if query.device != key.device or query.dtype != key.dtype:
            raise ValueError("paired ROCm RoPE Q/K must share one device and dtype")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("paired ROCm RoPE Q/K must share one head dimension")
        query_out, key_out = _RocmRoPEPairFunction.apply(
            query,
            key,
            positions,
            1_000_000.0,
        )
        if not query.requires_grad:
            query_out = query_out.detach()
        if not key.requires_grad:
            key_out = key_out.detach()
        return query_out, key_out

    @torch.inference_mode()
    def inference_direct() -> tuple[torch.Tensor, torch.Tensor]:
        return operator.forward_pair(query, key, positions)

    for _ in range(warmup):
        autograd_wrapper()
        inference_direct()
    torch.cuda.synchronize()

    baseline_out = autograd_wrapper()
    candidate_out = inference_direct()
    repeated_out = inference_direct()
    torch.cuda.synchronize()
    exact = all(
        torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        and torch.equal(actual.view(torch.uint8), repeated.view(torch.uint8))
        for expected, actual, repeated in zip(
            baseline_out,
            candidate_out,
            repeated_out,
            strict=True,
        )
    )
    if not exact:
        raise AssertionError("direct inference RoPE differs from the autograd baseline")

    samples: dict[str, list[float]] = {"autograd_wrapper": [], "inference_direct": []}
    functions = {
        "autograd_wrapper": autograd_wrapper,
        "inference_direct": inference_direct,
    }
    for block in range(blocks):
        order = (
            ("autograd_wrapper", "inference_direct")
            if block % 2 == 0
            else ("inference_direct", "autograd_wrapper")
        )
        for name in order:
            samples[name].append(_elapsed_ms(functions[name], iterations))

    baseline_ms = statistics.median(samples["autograd_wrapper"])
    candidate_ms = statistics.median(samples["inference_direct"])
    return {
        "tokens": tokens,
        "query_heads": query_heads,
        "key_heads": key_heads,
        "head_dim": head_dim,
        "autograd_wrapper_ms": baseline_ms,
        "inference_direct_ms": candidate_ms,
        "latency_reduction_percent": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
        "speedup": baseline_ms / candidate_ms,
        "raw_bytes_equal": exact,
        "query_sha256": _digest(candidate_out[0]),
        "key_sha256": _digest(candidate_out[1]),
        "autograd_wrapper_samples_ms": samples["autograd_wrapper"],
        "inference_direct_samples_ms": samples["inference_direct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,2,32")
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--key-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires ROCm PyTorch and a visible GPU")
    results = [
        _benchmark_tokens(
            tokens=int(tokens),
            query_heads=args.query_heads,
            key_heads=args.key_heads,
            head_dim=args.head_dim,
            warmup=args.warmup,
            blocks=args.blocks,
            iterations=args.iterations,
        )
        for tokens in args.tokens.split(",")
    ]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "dtype": "bfloat16",
                "blocks": args.blocks,
                "iterations_per_block": args.iterations,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
