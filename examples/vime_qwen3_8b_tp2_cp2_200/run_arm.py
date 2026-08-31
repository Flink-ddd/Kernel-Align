# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Submit one append-only VIME × RL-Kernel experiment arm to Ray."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Arm:
    group: str
    framework_use_rollout_logprobs: bool
    attention_case: str
    ffn_case: str
    logp_case: str
    description: str


ARMS = {
    "G00": Arm("G00", False, "P/P", "P/P", "P/P", "native VIME baseline"),
    "G10": Arm(
        "G10", True, "P/P", "P/P", "P/P", "VIME framework-level consistency only"
    ),
    "G01": Arm(
        "G01", False, "R/R", "R/R", "R/R", "RL-Kernel operator-level consistency only"
    ),
    "G11": Arm(
        "G11",
        True,
        "R/R",
        "R/R",
        "R/R",
        "framework-level and operator-level consistency",
    ),
}

MODEL_ARGS = (
    "--swiglu",
    "--num-layers",
    "36",
    "--hidden-size",
    "4096",
    "--ffn-hidden-size",
    "12288",
    "--num-attention-heads",
    "32",
    "--group-query-attention",
    "--num-query-groups",
    "8",
    "--use-rotary-position-embeddings",
    "--disable-bias-linear",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-6",
    "--rotary-base",
    "1000000",
    "--vocab-size",
    "151936",
    "--kv-channels",
    "128",
    "--qk-layernorm",
    "--untie-embeddings-and-output-weights",
)


def _path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"{label} is required (argument or environment variable)")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_state(path: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_diff = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "revision": _revision(path),
        "dirty": bool(status),
        "status": status,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _gpu_inventory() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = []
    for line in result.stdout.splitlines():
        index, name, memory, driver = (field.strip() for field in line.split(",", 3))
        inventory.append(
            {"index": index, "name": name, "memory_mib": memory, "driver": driver}
        )
    return inventory


def _default_run_id(group: str, num_rollout: int, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{group.lower()}-n{num_rollout}-s{seed}-{stamp}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(ARMS), required=True)
    parser.add_argument("--num-rollout", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rollout-seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rl-kernel-root", default=os.environ.get("RL_KERNEL_ROOT"))
    parser.add_argument("--vime-root", default=os.environ.get("VIME_ROOT"))
    parser.add_argument("--megatron-root", default=os.environ.get("MEGATRON_ROOT"))
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT"))
    parser.add_argument("--ref-load", default=os.environ.get("TORCH_DIST_ROOT"))
    parser.add_argument("--prompt-data", default=os.environ.get("PROMPT_DATA"))
    parser.add_argument(
        "--python", default=os.environ.get("RL_KERNEL_REAL_PYTHON", sys.executable)
    )
    parser.add_argument("--ray-bin", default=os.environ.get("RAY_BIN"))
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_API_SERVER_ADDRESS", "http://127.0.0.1:8265"),
    )
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--n-samples-per-prompt", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--max-response-len", type=int, default=7168)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=4096)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--extra-pythonpath", action="append", default=[])
    parser.add_argument(
        "--ld-library-path", default=os.environ.get("LD_LIBRARY_PATH", "")
    )
    parser.add_argument(
        "--wait", action="store_true", help="stream logs until the Ray job exits"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit development runs from dirty repository states",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="write the manifest without submitting"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_rollout <= 0:
        raise ValueError("--num-rollout must be positive")
    arm = ARMS[args.group]

    script_dir = Path(__file__).resolve().parent
    rl_kernel_root = _path(args.rl_kernel_root, "RL-Kernel root")
    vime_root = _path(args.vime_root, "VIME root")
    megatron_root = _path(args.megatron_root, "Megatron root")
    model_root = _path(args.model_root, "HF model root")
    ref_load = _path(args.ref_load, "Megatron torch-dist checkpoint")
    prompt_data = _path(args.prompt_data, "prompt data")
    python = _path(args.python, "Python executable")
    ray_bin = _path(args.ray_bin or str(python.parent / "ray"), "Ray executable")
    entrypoint = _path(
        str(script_dir / "aligned_python_entrypoint.sh"), "aligned Python entrypoint"
    )
    repository_state = {
        "rl_kernel": _repository_state(rl_kernel_root),
        "vime": _repository_state(vime_root),
        "megatron": _repository_state(megatron_root),
    }
    dirty_repositories = [
        name for name, state in repository_state.items() if state["dirty"]
    ]
    if dirty_repositories and not args.allow_dirty:
        raise RuntimeError(
            "refusing a non-reproducible run from dirty repositories: "
            + ", ".join(dirty_repositories)
            + "; commit the changes or pass --allow-dirty for a development run"
        )

    run_id = args.run_id or _default_run_id(args.group, args.num_rollout, args.seed)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dot, underscore, and dash"
        )
    run_dir = args.output_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    pythonpath = [str(rl_kernel_root), str(vime_root), str(megatron_root)]
    pythonpath.extend(
        str(Path(item).expanduser().resolve()) for item in args.extra_pythonpath
    )
    if os.environ.get("PYTHONPATH"):
        pythonpath.extend(
            item for item in os.environ["PYTHONPATH"].split(os.pathsep) if item
        )

    env_vars = {
        "RL_KERNEL_ROOT": str(rl_kernel_root),
        "RL_KERNEL_REAL_PYTHON": str(python),
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(pythonpath)),
        "LD_LIBRARY_PATH": args.ld_library_path,
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": str(args.seed),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        "CUBLASLT_WORKSPACE_SIZE": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "RL_KERNEL_VLLM_INTEGRATION": "1",
        "RL_KERNEL_CUDA_ONLY": "1",
        "VIME_RL_KERNEL_STRICT": "1",
        "RL_KERNEL_ATTENTION_CASE": arm.attention_case,
        "RL_KERNEL_FFN_CASE": arm.ffn_case,
        "RL_KERNEL_LOGP_CASE": arm.logp_case,
        "RL_KERNEL_READBACK_DIR": str(run_dir / "readbacks"),
        "RL_KERNEL_VLLM_REAL_VOCAB_SIZE": "151936",
        "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE": "152064",
        "RL_KERNEL_VLLM_TEMPERATURE": "1.0",
        "RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE": str(
            args.rollout_batch_size * args.n_samples_per_prompt
        ),
        "RL_KERNEL_SEED": str(args.seed),
        "RL_KERNEL_ROLLOUT_SEED": str(args.rollout_seed),
        "RL_KERNEL_RUN_ID": run_id,
    }
    if os.environ.get("CUDNN_FRONTEND_CUDART_LIB_NAME"):
        env_vars["CUDNN_FRONTEND_CUDART_LIB_NAME"] = os.environ[
            "CUDNN_FRONTEND_CUDART_LIB_NAME"
        ]

    train_command = [
        str(entrypoint),
        "train.py",
        "--train-backend",
        "megatron",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "4",
        "--rollout-num-gpus",
        "4",
        *MODEL_ARGS,
        "--hf-checkpoint",
        str(model_root),
        "--ref-load",
        str(ref_load),
        "--load",
        str(run_dir / "initial-load"),
        "--prompt-data",
        str(prompt_data),
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--rm-type",
        "deepscaler",
        "--advantage-estimator",
        "grpo",
        "--num-rollout",
        str(args.num_rollout),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--n-samples-per-prompt",
        str(args.n_samples_per_prompt),
        "--rollout-max-response-len",
        str(args.max_response_len),
        "--rollout-temperature",
        "1.0",
        "--rollout-top-p",
        "1.0",
        "--global-batch-size",
        str(args.global_batch_size),
        "--balance-data",
        "--tensor-model-parallel-size",
        "2",
        "--context-parallel-size",
        "2",
        "--cp-comm-type",
        "p2p",
        "--pipeline-model-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--expert-tensor-parallel-size",
        "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        str(args.max_tokens_per_gpu),
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--custom-megatron-init-path",
        "rl_engine.integrations.megatron_runtime.initialize_from_environment",
        "--save-debug-train-data",
        str(run_dir / "train-data" / "{rollout_id}.rank{rank}.pt"),
        "--update-weight-mode",
        "full",
        "--update-weight-transport",
        "disk",
        "--update-weight-disk-dir",
        str(run_dir / "weight-updates"),
        "--linear-logp-provider",
        "rl_engine.integrations.vime.linear_logp_provider.provider",
        "--linear-logp-provider-mode",
        "strict",
        "--no-save-optim",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--transformer-impl",
        "transformer_engine",
        "--no-persist-layer-norm",
        "--no-gradient-accumulation-fusion",
        "--no-rope-fusion",
        "--attention-softmax-in-fp32",
        "--attention-backend",
        "auto",
        "--router-policy",
        "cache_aware",
        "--rollout-num-gpus-per-engine",
        "2",
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
    ]
    if arm.framework_use_rollout_logprobs:
        train_command.append("--use-rollout-logprobs")
    if {arm.attention_case, arm.ffn_case, arm.logp_case} == {"R/R"}:
        train_command.extend(
            [
                "--ci-test",
                "--ci-disable-kl-checker",
                "--ci-train-rollout-logprob-abs-diff-threshold",
                "0",
            ]
        )

    submission_id = f"vime200-{run_id}"
    runtime_env = {"env_vars": env_vars}
    ray_command = [
        str(ray_bin),
        "job",
        "submit",
        f"--address={args.ray_address}",
        "--submission-id",
        submission_id,
        "--runtime-env-json",
        json.dumps(runtime_env, separators=(",", ":")),
        "--metadata-json",
        json.dumps({"group": args.group, "run_id": run_id}),
        "--working-dir",
        str(vime_root),
    ]
    if not args.wait:
        ray_command.append("--no-wait")
    ray_command.extend(["--", *train_command])

    manifest = {
        "schema_version": "rlkernel.vime_qwen3_8b_tp2_cp2_200.run.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if args.dry_run else "submitting",
        "run_id": run_id,
        "submission_id": submission_id,
        "arm": asdict(arm),
        "num_rollout": args.num_rollout,
        "seed": args.seed,
        "rollout_seed": args.rollout_seed,
        "topology": {
            "gpus": 8,
            "actor_gpus": 4,
            "rollout_gpus": 4,
            "tp": 2,
            "cp": 2,
            "pp": 1,
        },
        "batching": {
            "rollout_batch_size": args.rollout_batch_size,
            "n_samples_per_prompt": args.n_samples_per_prompt,
            "global_batch_size": args.global_batch_size,
            "max_response_len": args.max_response_len,
            "max_tokens_per_gpu": args.max_tokens_per_gpu,
        },
        "algorithm": {
            "advantage_estimator": "grpo",
            "reward_model": "deepscaler",
        },
        "training_memory": {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
        "vllm_execution": {
            "cudagraph_required": True,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "capture_sizes": list(
                range(1, args.rollout_batch_size * args.n_samples_per_prompt + 1)
            ),
            "enforce_eager": False,
        },
        "paths": {
            "run_dir": str(run_dir),
            "rl_kernel_root": str(rl_kernel_root),
            "vime_root": str(vime_root),
            "megatron_root": str(megatron_root),
            "model_root": str(model_root),
            "ref_load": str(ref_load),
            "prompt_data": str(prompt_data),
        },
        "revisions": {
            name: state["revision"] for name, state in repository_state.items()
        },
        "repository_state": repository_state,
        "prompt_data_sha256": _sha256(prompt_data),
        "gpu_inventory": _gpu_inventory(),
        "runtime_env": runtime_env,
        "train_command": train_command,
        "ray_command": ray_command,
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "run_dir": str(run_dir),
                    "manifest": str(manifest_path),
                }
            )
        )
        return 0

    result = subprocess.run(ray_command, capture_output=True, text=True)
    manifest["submission"] = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    manifest["status"] = "submitted" if result.returncode == 0 else "submission_failed"
    _write_json(manifest_path, manifest)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "submission_id": submission_id,
                "run_dir": str(run_dir),
            }
        )
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
