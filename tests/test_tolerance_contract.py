# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import hashlib
import inspect
import json

import pytest
import torch

from rl_engine.kernels.gtest.tolerance import (
    load_contract,
    resolve_logprob_threshold,
    tolerance_contract_fingerprint,
)


def test_load_contract_contains_expected_operator_classes():
    contract = load_contract()
    accuracy = contract["accuracy"]["default"]
    assert set(accuracy) == {"elementwise", "reduction", "logprob"}


def test_load_contract_contains_expected_dtypes():
    contract = load_contract()
    for op_class in ("elementwise", "reduction", "logprob"):
        assert set(contract["accuracy"]["default"][op_class]) == {
            "float32",
            "bfloat16",
            "float16",
        }


def test_logprob_bfloat16_tolerance_covers_observed_reference_drift():
    contract = load_contract()
    tolerance = contract["accuracy"]["default"]["logprob"]["bfloat16"]
    assert tolerance["atol"] >= 5.0e-2
    assert tolerance["rtol"] == 0.0


@pytest.mark.parametrize(
    ("dtype", "dtype_name"),
    (
        (torch.float32, "float32"),
        ("fp32", "float32"),
        (torch.bfloat16, "bfloat16"),
        ("bf16", "bfloat16"),
        (torch.float16, "float16"),
        ("fp16", "float16"),
    ),
)
def test_resolve_logprob_threshold_reads_current_ws1_absolute_tolerance(dtype, dtype_name):
    expected = load_contract()["accuracy"]["default"]["logprob"][dtype_name]["atol"]

    assert resolve_logprob_threshold(dtype) == expected


def test_resolve_logprob_threshold_has_no_contract_or_value_override_parameter():
    assert tuple(inspect.signature(resolve_logprob_threshold).parameters) == ("dtype",)


def test_resolve_logprob_threshold_rejects_dtype_outside_ws1_contract():
    with pytest.raises(ValueError, match="unsupported WS1 logprob dtype"):
        resolve_logprob_threshold(torch.float64)


def test_tolerance_contract_fingerprint_is_canonical_content_sha256():
    canonical = json.dumps(
        load_contract(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert tolerance_contract_fingerprint() == hashlib.sha256(canonical).hexdigest()
    assert len(tolerance_contract_fingerprint()) == 64
