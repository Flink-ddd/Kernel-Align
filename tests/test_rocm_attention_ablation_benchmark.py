# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "benchmarks" / "benchmark_rocm_attention_ablation.py"
CHECKED_IN_RESULT = ROOT / "benchmarks" / "results" / "pr230_rocm_mi300x_ablation" / "results.json"
SPEC = importlib.util.spec_from_file_location("rocm_attention_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _metric(*, drift: bool, shape: list[int], dtype: str):
    return {
        "max_abs": 0.5 if drift else 0.0,
        "mismatch_count": 1 if drift else 0,
        "element_count": math.prod(shape),
        "bitwise_equal": not drift,
        "left_dtype": dtype,
        "right_dtype": dtype,
        "shape": shape,
    }


def _valid_payload():
    shapes = [list(shape) for shape in MODULE.DEFAULT_SHAPES]
    cases = []
    for batch, sequence in MODULE.DEFAULT_SHAPES:
        for matrix_row in MODULE.attention_debug_matrix()["rows"]:
            row_id = matrix_row["id"]
            common = {
                "row_id": row_id,
                "batch": batch,
                "sequence": sequence,
                "category": matrix_row["category"],
                "probe": matrix_row["probe"],
                "expected": matrix_row["expected"],
                "passed": True,
                "realization": MODULE.ROW_REALIZATIONS[row_id],
            }
            if row_id == "A4":
                binding = MODULE._topology_gate(batch=batch, sequence=sequence)
                cases.append(
                    {
                        **common,
                        "comparable": False,
                        "outcome": "rejected",
                        "gate_implementation": (
                            "rl_engine.alignment.cross_config.bind_attention_contracts"
                        ),
                        "identity_errors": [issue["field"] for issue in binding["issues"]],
                        "binding_gate": binding,
                        "metrics": {name: None for name in MODULE.METRIC_NAMES},
                    }
                )
                continue
            drift = matrix_row["expected"] == "diagnostic"
            implementations = MODULE.ROW_IMPLEMENTATIONS[row_id]
            metric_shapes = {
                "out": [
                    batch,
                    MODULE.QWEN3_Q_HEADS,
                    1 if row_id == "C2" else sequence,
                    MODULE.QWEN3_HEAD_DIM,
                ],
                "lse": [batch, MODULE.QWEN3_Q_HEADS, 1 if row_id == "C2" else sequence],
                "dq": [batch, MODULE.QWEN3_Q_HEADS, sequence, MODULE.QWEN3_HEAD_DIM],
                "dk": [batch, MODULE.QWEN3_KV_HEADS, sequence, MODULE.QWEN3_HEAD_DIM],
                "dv": [batch, MODULE.QWEN3_KV_HEADS, sequence, MODULE.QWEN3_HEAD_DIM],
            }
            cases.append(
                {
                    **common,
                    "comparable": True,
                    "outcome": "drift_detected" if drift else "matched",
                    "implementations": {
                        "baseline": implementations[0],
                        "candidate": implementations[1],
                    },
                    "metrics": {
                        name: _metric(
                            drift=drift,
                            shape=metric_shapes[name],
                            dtype="torch.float32" if name == "lse" else "torch.bfloat16",
                        )
                        for name in MODULE.METRIC_NAMES
                    },
                }
            )
    return {
        "schema_version": MODULE.RESULT_SCHEMA,
        "scope": MODULE.RESULT_SCOPE,
        "command": ["python", "benchmarks/benchmark_rocm_attention_ablation.py"],
        "source_provenance": {
            "revision": "a" * 40,
            "tracked_dirty": False,
            "tracked_diff_sha256": None,
            "script_path": "benchmarks/benchmark_rocm_attention_ablation.py",
            "script_sha256": "b" * 64,
            "script_matches_head": True,
        },
        "matrix_manifest": MODULE.attention_debug_matrix(),
        "environment": {
            "python": "3.10.0",
            "pytorch": "2.12.0+rocm7.0",
            "hip_runtime": "7.0",
            "device_index": 0,
            "device_name": "AMD Instinct MI300X",
            "architecture": "gfx942:sramecc+:xnack-",
            "gpu_count": 1,
            "primary_backend_id": MODULE.ROCM_REFERENCE_BACKEND_ID,
            "primary_core_id": MODULE.STRICT_ATTENTION_REFERENCE_CORE_ID,
            "primary_schedule": MODULE.STRICT_ATTENTION_SCHEDULE_ID,
            "primary_reference_only": True,
            "primary_production_ready": False,
            "execution_kind": "operator_only_rocm_reference",
        },
        "configuration": {
            "seed": 230,
            "dtype": "bfloat16",
            "q_heads": MODULE.QWEN3_Q_HEADS,
            "kv_heads": MODULE.QWEN3_KV_HEADS,
            "head_dim": MODULE.QWEN3_HEAD_DIM,
            "shapes": shapes,
        },
        "cases": cases,
        "matrix": MODULE._aggregate(cases),
    }


def test_metric_records_bitwise_and_numerical_drift():
    same = MODULE._metric(torch.tensor([1.0]), torch.tensor([1.0]))
    drift = MODULE._metric(torch.tensor([1.0]), torch.tensor([1.5]))

    assert same == {
        "max_abs": 0.0,
        "mismatch_count": 0,
        "element_count": 1,
        "bitwise_equal": True,
        "left_dtype": "torch.float32",
        "right_dtype": "torch.float32",
        "shape": [1],
    }
    assert drift["max_abs"] == 0.5
    assert drift["mismatch_count"] == 1
    assert drift["bitwise_equal"] is False


def test_metric_uses_raw_bits_and_requires_matching_dtype():
    signed_zero = MODULE._metric(torch.tensor([0.0]), torch.tensor([-0.0]))
    mixed_dtype = MODULE._metric(
        torch.tensor([1.0], dtype=torch.bfloat16),
        torch.tensor([1.0], dtype=torch.float32),
    )

    assert signed_zero["max_abs"] == 0.0
    assert signed_zero["mismatch_count"] == 1
    assert signed_zero["bitwise_equal"] is False
    assert mixed_dtype["mismatch_count"] == 1
    assert mixed_dtype["left_dtype"] != mixed_dtype["right_dtype"]


def test_chunk_merge_probe_has_attention_shapes_and_finite_values():
    generator = torch.Generator().manual_seed(230)
    q = torch.randn(1, 4, 8, 128, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(1, 1, 8, 128, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(1, 1, 8, 128, dtype=torch.bfloat16, generator=generator)
    dout = torch.randn(q.shape, dtype=torch.bfloat16, generator=generator)
    dense = MODULE._evaluate(
        q,
        k,
        v,
        dout,
        MODULE._dense_attention(accumulator_dtype=torch.float32),
    )

    for order in ("ascending", "descending"):
        chunked = MODULE._evaluate(q, k, v, dout, MODULE._chunked_attention(order))
        assert chunked.out.shape == q.shape
        assert chunked.lse.shape == q.shape[:-1]
        assert all(torch.isfinite(getattr(chunked, name)).all() for name in MODULE.METRIC_NAMES)
        maximums = {
            name: MODULE._metric(getattr(dense, name), getattr(chunked, name))["max_abs"]
            for name in MODULE.METRIC_NAMES
        }
        assert maximums["lse"] <= 1.0e-5
        assert all(maximums[name] <= 0.015625 for name in ("out", "dq", "dk", "dv"))


def test_explicit_accumulator_probe_changes_all_five_metrics():
    generator = torch.Generator().manual_seed(231)
    q = torch.randn(1, 4, 4, 128, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16, generator=generator)
    v = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16, generator=generator)
    dout = torch.randn(q.shape, dtype=torch.bfloat16, generator=generator)
    fp32 = MODULE._evaluate(q, k, v, dout, MODULE._dense_attention(accumulator_dtype=torch.float32))
    bf16 = MODULE._evaluate(
        q, k, v, dout, MODULE._dense_attention(accumulator_dtype=torch.bfloat16)
    )

    assert all(
        MODULE._metric(getattr(fp32, name), getattr(bf16, name))["mismatch_count"] > 0
        for name in MODULE.METRIC_NAMES
    )


def test_topology_probe_uses_binding_gate_and_isolates_ownership_mismatch():
    binding = MODULE._topology_gate(batch=2, sequence=32)

    assert binding["comparable"] is False
    assert binding["passed"] is False
    assert {issue["code"] for issue in binding["issues"]} == {"TOPOLOGY_MISMATCH"}
    assert {issue["field"] for issue in binding["issues"]} == {
        "sharding.tp_rank",
        "sharding.local_q_head_start",
        "sharding.local_kv_head_start",
    }


def test_payload_validator_accepts_complete_pr230_rocm_evidence():
    payload = _valid_payload()
    MODULE.validate_payload(payload)


def _invent_a4_issue(payload):
    case = payload["cases"][4]
    case["binding_gate"]["issues"] = [
        {
            "code": "TOPOLOGY_MISMATCH",
            "tier": "identical",
            "field": "invented",
            "rollout": 1,
            "training": 0,
        }
    ]
    case["identity_errors"] = ["invented"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(cases=[]), "cover every PR230 row"),
        (
            lambda payload: payload["cases"][1]["metrics"]["out"].update(max_abs=float("nan")),
            "finite and non-negative",
        ),
        (
            lambda payload: payload["cases"][4]["binding_gate"].update(comparable=True),
            "topology-gate rejection",
        ),
        (_invent_a4_issue, "topology-gate rejection"),
        (
            lambda payload: payload["cases"][0]["metrics"]["out"].update(
                shape=[1], element_count=1
            ),
            "incompatible dtype or shape",
        ),
        (
            lambda payload: payload["configuration"]["shapes"].pop(),
            "exact eight-shape",
        ),
        (
            lambda payload: payload["configuration"].pop("seed"),
            "configuration.seed",
        ),
        (
            lambda payload: payload["matrix"][0].update(case_count=0),
            "does not reproduce",
        ),
        (
            lambda payload: payload["environment"].update(architecture="sm_90"),
            "gfx942 ROCm",
        ),
        (
            lambda payload: payload["environment"].pop("device_name"),
            "environment.device_name",
        ),
    ],
)
def test_payload_validator_rejects_incomplete_or_fabricated_evidence(mutate, message):
    payload = copy.deepcopy(_valid_payload())
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        MODULE.validate_payload(payload)


def test_repository_provenance_rejects_unbacked_hashes():
    with pytest.raises(ValueError, match="not backed"):
        MODULE.validate_repository_provenance(_valid_payload())


def test_checked_in_mi300x_matrix_is_complete_and_source_backed():
    payload = json.loads(CHECKED_IN_RESULT.read_text(encoding="utf-8"))

    MODULE.validate_payload(payload)
    MODULE.validate_repository_provenance(payload)
    assert all(row["passed"] for row in payload["matrix"])
