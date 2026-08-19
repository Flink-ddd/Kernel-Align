# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Run the ROCm strict Attention acceptance matrix and retain evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/rocm-attention"))
    args = parser.parse_args()
    if torch.version.hip is None or torch.cuda.device_count() < 8:
        raise RuntimeError("the formal acceptance matrix requires 8 visible ROCm GPUs")

    repo = Path(__file__).resolve().parents[1]
    output = (
        (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    )
    output.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, object]] = []

    commands = [
        (
            "single_gpu",
            [sys.executable, "-m", "pytest", "-q", "tests/test_deterministic_attention_rocm.py"],
        )
    ]
    for transport, strict in (("p2p_nccl_reference", False), ("rccl_ag_rs", True)):
        for ranks in (2, 4, 8):
            name = f"{transport}_{ranks}r"
            report = output / f"{name}.json"
            command = [
                "torchrun",
                "--standalone",
                f"--nproc-per-node={ranks}",
                "scripts/ws2_p2p_nccl_attention_reference_check.py",
                "--transport",
                transport,
                "--output",
                str(report),
            ]
            if strict:
                command.append("--strict-shared-core")
            commands.append((name, command))

    for name, command in commands:
        completed = subprocess.run(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (output / f"{name}.log").write_text(completed.stdout, encoding="utf-8")
        steps.append({"name": name, "command": command, "returncode": completed.returncode})

    summary = {
        "schema_version": "ws2_rocm_attention_acceptance/v1",
        "platform": "rocm",
        "torch": str(torch.__version__),
        "hip": str(torch.version.hip),
        "collective": list(torch.cuda.nccl.version()),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "steps": steps,
        "passed": all(step["returncode"] == 0 for step in steps),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
