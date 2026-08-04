#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.testing import LogprobComparisonInputs, compare_single_gpu_logprob  # noqa: E402
from rl_engine.utils.logger import logger  # noqa: E402


def _dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _route_rl_kernel_logs_to_stderr() -> None:
    """Keep stdout machine-readable while preserving backend diagnostics."""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the WS2 TP=1 selected-logprob/LSE comparison harness."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=("pytorch", "triton", "cuda-sm90"),
        help="Exact backend to compare. Repeat for multiple backends; defaults to pytorch.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=257)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    _route_rl_kernel_logs_to_stderr()
    args = parse_args()
    device = _device(args.device)
    if args.batch < 1 or args.seq < 1 or args.vocab < 1:
        raise ValueError("batch, seq, and vocab must be positive")
    if not 0 <= args.prompt_tokens <= args.seq:
        raise ValueError("prompt-tokens must be in [0, seq]")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    logits = torch.randn(
        args.batch,
        args.seq,
        args.vocab,
        generator=generator,
        device=device,
        dtype=_dtype(args.dtype),
    )
    target_ids = torch.randint(
        0,
        args.vocab,
        (args.batch, args.seq),
        generator=generator,
        device=device,
    )
    active_mask = torch.ones((args.batch, args.seq), device=device, dtype=torch.bool)
    active_mask[:, : args.prompt_tokens] = False
    report = compare_single_gpu_logprob(
        LogprobComparisonInputs(
            logits=logits,
            target_ids=target_ids,
            active_token_mask=active_mask,
        ),
        candidates=tuple(args.candidate or ("pytorch",)),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
