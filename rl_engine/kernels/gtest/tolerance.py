# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CONTRACT_PATH = Path(__file__).with_name("tolerance_contract.json")


def load_contract(path: str | Path = _CONTRACT_PATH) -> dict[str, Any]:
    """Load the dtype/operator-class tolerance contract."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_logprob_threshold(dtype: Any) -> float:
    """Return the fixed WS1 selected-logprob absolute-difference threshold.

    The contract path is intentionally not configurable through this accessor.
    Cross-configuration experiment definitions may select a dtype, but they cannot
    inject or override a numerical threshold.
    """

    dtype_name = _normalize_dtype_name(dtype)
    contract = load_contract()
    try:
        values = contract["accuracy"]["default"]["logprob"][dtype_name]
        return float(values["atol"])
    except KeyError as exc:
        raise ValueError(f"WS1 has no logprob threshold for dtype {dtype_name!r}") from exc


def tolerance_contract_fingerprint() -> str:
    """Return a deterministic fingerprint of the current WS1 contract contents."""

    canonical = json.dumps(
        load_contract(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalize_dtype_name(dtype: Any) -> str:
    normalized = str(dtype).strip().lower().replace("torch.", "").replace("-", "")
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
        "float": "float32",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"unsupported WS1 logprob dtype {dtype!r}; expected one of: {valid}"
        ) from exc


__all__ = [
    "load_contract",
    "resolve_logprob_threshold",
    "tolerance_contract_fingerprint",
]
