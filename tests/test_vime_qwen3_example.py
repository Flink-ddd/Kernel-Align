# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.vime_qwen3_8b_tp2_cp2.run import (
    build_report,
    load_config,
    validate_config,
    validate_runtime_evidence,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "examples" / "vime_qwen3_8b_tp2_cp2" / "qwen3_8b_tp2_cp2.json"


def test_qwen3_example_config_is_strict_and_explicit():
    config = load_config(CONFIG)
    validate_config(config)
    assert config["training"]["tensor_model_parallel_size"] == 2
    assert config["training"]["context_parallel_size"] == 2
    assert config["selected_logprob_provider"]["mode"] == "strict"


def test_qwen3_example_report_does_not_claim_unread_back_attention_or_ffn(tmp_path):
    config = load_config(CONFIG)
    report = build_report(
        config,
        vime_root=tmp_path / "vime",
        rl_kernel_root=tmp_path / "rl-kernel",
        command=["bash", "run.sh"],
        status="passed",
        returncode=0,
        log_text="Selected-logprob provider active: backend_id=pytorch-vocab-parallel-logp-ws2",
        log_path=tmp_path / "run.log",
    )
    assert report["status"] == "passed"
    assert report["claim_boundary"]["qwen3_8b_tp2_cp2_vime_training"] is True
    assert report["claim_boundary"]["attention_train_infer_consistency"] == "unclaimed"
    assert report["claim_boundary"]["ffn_train_infer_consistency"] == "unclaimed"
    assert report["provider"]["fallback_observed"] is False


def test_qwen3_example_fails_closed_when_provider_marker_is_missing(tmp_path):
    config = load_config(CONFIG)
    report = build_report(
        config,
        vime_root=tmp_path / "vime",
        rl_kernel_root=tmp_path / "rl-kernel",
        command=["bash", "run.sh"],
        status="passed",
        returncode=0,
        log_text="training completed without provider provenance",
        log_path=None,
    )
    assert report["status"] == "failed"
    assert report["claim_boundary"]["qwen3_8b_tp2_cp2_vime_training"] is False


def _runtime_evidence():
    return {
        "schema_version": "rlkernel.operator_runtime_evidence.v1",
        "operators": {
            "attention": {
                "training": {"implementation_id": "rlk.attn", "backend_id": "rlk", "contract_id": "a"},
                "rollout": {"implementation_id": "rlk.attn", "backend_id": "rlk", "contract_id": "a"},
                "comparison": {
                    "passed": True,
                    "out_max_abs": 0.0,
                    "lse_max_abs": 0.0,
                    "dq_max_abs": 0.0,
                    "dk_max_abs": 0.0,
                    "dv_max_abs": 0.0,
                },
            },
            "ffn": {
                "training": {"implementation_id": "rlk.ffn", "backend_id": "rlk", "contract_id": "f"},
                "rollout": {"implementation_id": "rlk.ffn", "backend_id": "rlk", "contract_id": "f"},
                "comparison": {
                    "passed": True,
                    "out_max_abs": 0.0,
                    "dx_max_abs": 0.0,
                    "dw_max_abs": 0.0,
                },
            },
        },
    }


def test_qwen3_example_accepts_only_exact_zero_runtime_evidence(tmp_path):
    evidence = _runtime_evidence()
    validate_runtime_evidence(evidence)
    config = load_config(CONFIG)
    report = build_report(
        config,
        vime_root=tmp_path / "vime",
        rl_kernel_root=tmp_path / "rl-kernel",
        command=["bash", "run.sh"],
        status="passed",
        returncode=0,
        log_text="Selected-logprob provider active: backend_id=pytorch-vocab-parallel-logp-ws2",
        log_path=None,
        runtime_evidence=evidence,
    )
    assert report["claim_boundary"]["attention_train_infer_consistency"] == "passed"
    assert report["claim_boundary"]["ffn_train_infer_consistency"] == "passed"


def test_qwen3_example_rejects_nonzero_runtime_evidence():
    evidence = _runtime_evidence()
    evidence["operators"]["attention"]["comparison"]["out_max_abs"] = 1e-6
    with pytest.raises(ValueError, match="attention"):
        validate_runtime_evidence(evidence)


@pytest.mark.parametrize("bad_path", ["", "other.provider"])
def test_qwen3_example_rejects_non_rlkernel_provider(bad_path):
    config = load_config(CONFIG)
    config["selected_logprob_provider"]["path"] = bad_path
    with pytest.raises(ValueError, match="RL-Kernel Vime provider"):
        validate_config(config)
