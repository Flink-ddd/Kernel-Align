# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from dataclasses import asdict
import os
from pathlib import Path
import subprocess

from examples.vime_qwen3_8b_tp4_cp2_200.run_arm import (
    ARMS,
    MEGATRON_ATTENTION_BACKEND,
    RL_KERNEL_LINEAR_LOGP_PROVIDER,
    _linear_logp_provider_args,
)
from examples.vime_qwen3_8b_tp4_cp2_200.validate_run import (
    VIME_NATIVE_LINEAR_LOGP_MARKER,
    _validate_readbacks,
)


def _production_operator(framework: str, target: str, module: str) -> dict:
    return {
        "framework": framework,
        "target": target,
        "module": module,
        "case_id": "P/P",
        "implementation": "production",
        "backend_id": f"{framework}.production.{module}",
        "call_count": 1,
        "provenance": {"runtime_platform": "cuda"},
    }


def _production_readbacks() -> list[dict]:
    megatron_modules = ("attention", "ffn")
    vllm_modules = ("attention", "ffn", "logp")
    return [
        {
            "framework": "megatron",
            "target": "training",
            "installed_hooks": {module: module for module in megatron_modules},
            "operators": {
                module: _production_operator("megatron", "training", module)
                for module in megatron_modules
            },
            "fallbacks": [],
        },
        {
            "framework": "vllm",
            "target": "rollout",
            "installed_hooks": {module: module for module in vllm_modules},
            "operators": {
                module: _production_operator("vllm", "rollout", module)
                for module in vllm_modules
            },
            "fallbacks": [],
        },
    ]


def test_tp4_formal_matrix_pins_the_vime_qwen3_attention_backend():
    assert MEGATRON_ATTENTION_BACKEND == "fused"


def test_launcher_forces_cuda_graph_without_a_logp_provider():
    root = Path(__file__).parents[1]
    launcher = (
        root
        / "examples"
        / "vime_qwen3_8b_tp4_cp2_200"
        / "aligned_python_entrypoint.sh"
    )
    env = os.environ.copy()
    env.update(
        RL_KERNEL_REAL_PYTHON="/bin/echo",
        RL_KERNEL_ROOT=str(root),
        RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE="8",
    )

    result = subprocess.run(
        [
            str(launcher),
            "train.py",
            "--rollout-batch-size",
            "1",
            "--n-samples-per-prompt",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "--linear-logp-provider" not in result.stdout
    assert "--vllm-optimization-level 0" in result.stdout
    assert '"cudagraph_mode":"FULL_DECODE_ONLY"' in result.stdout
    assert '"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8]' in result.stdout
    assert "required vLLM full-decode CUDA Graph capture sizes" in result.stderr


def test_production_arms_do_not_install_the_rlkernel_logp_provider():
    assert _linear_logp_provider_args(ARMS["G00"]) == ()
    assert _linear_logp_provider_args(ARMS["G10"]) == ()


def test_rlkernel_arms_install_the_strict_logp_provider():
    expected = (
        "--linear-logp-provider",
        RL_KERNEL_LINEAR_LOGP_PROVIDER,
        "--linear-logp-provider-mode",
        "strict",
    )

    assert _linear_logp_provider_args(ARMS["G01"]) == expected
    assert _linear_logp_provider_args(ARMS["G11"]) == expected


def test_validator_accepts_native_vime_logp_evidence_for_production_arm():
    report = _validate_readbacks(
        _production_readbacks(),
        asdict(ARMS["G10"]),
        VIME_NATIVE_LINEAR_LOGP_MARKER,
    )

    assert report["passed"]
    training_logp = report["frameworks"]["megatron/training"]["modules"]["logp"]
    assert training_logp["native_marker_present"]
    assert training_logp["call_count"] == 0


def test_validator_rejects_provider_readback_on_production_megatron_logp():
    readbacks = _production_readbacks()
    contaminated = _production_operator("megatron", "training", "logp")
    contaminated["backend_id"] = "pytorch-vocab-parallel-logp-ws2"
    contaminated["provenance"] = {
        "runtime_platform": "cuda",
        "actual_backend": "rlkernel.linear_logp.bitwise.v1",
        "deterministic_linear_logp": True,
        "execution": {"strict_backend": True},
    }
    readbacks[0]["installed_hooks"]["logp"] = "rlkernel-provider"
    readbacks[0]["operators"]["logp"] = contaminated

    report = _validate_readbacks(
        readbacks,
        asdict(ARMS["G10"]),
        VIME_NATIVE_LINEAR_LOGP_MARKER,
    )

    assert not report["passed"]
    assert any("unexpectedly entered provider readback" in error for error in report["errors"])


def test_validator_rejects_production_label_over_rlkernel_actual_backend():
    readbacks = _production_readbacks()
    readbacks[1]["operators"]["logp"]["provenance"] = {
        "runtime_platform": "cuda",
        "actual_backend": "rlkernel.linear_logp.bitwise.v1",
    }

    report = _validate_readbacks(
        readbacks,
        asdict(ARMS["G10"]),
        VIME_NATIVE_LINEAR_LOGP_MARKER,
    )

    assert not report["passed"]
    assert any("production route executed an RL-Kernel backend" in error for error in report["errors"])
