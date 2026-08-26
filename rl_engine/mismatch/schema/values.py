# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Base value types. Imports nothing from the project.

Everything here is a plain data structure: public fields, ``frozen=True``, no
meaningful methods. Behaviour lives in free functions under ``pipeline/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class PolicyRole(str, Enum):
    """Which of the two compared policies this side plays.

    Not ``Policy``: in RL that means the policy itself, not the part it plays.
    """

    ROLLOUT = "rollout"  # pi_old -- produced the trajectory
    TRAINING = "training"  # pi_theta -- being updated


class ExecutionPath(str, Enum):
    """Which physical path produced a set of logprobs.

    Whole sequence at once, in chunks, or token by token: mathematically
    equivalent, not equal in floating point.
    """

    TRAINING_FULL_PREFILL = "training_full_prefill"
    ROLLOUT_FULL_PREFILL = "rollout_full_prefill"
    ROLLOUT_CHUNKED_PREFILL = "rollout_chunked_prefill"
    ROLLOUT_DECODE = "rollout_decode"


class Precision(str, Enum):
    FP64 = "fp64"  # oracle only
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


class DowncastPoint(str, Enum):
    """When a high-precision accumulator is written back at lower precision.

    Numerical precision reduction, not the C++ sense of casting down a hierarchy.
    """

    NEVER = "never"
    PER_BLOCK = "per_block"  # largest error
    PER_PARTIAL = "per_partial"
    FINAL_WRITE = "final_write"  # smallest error


class RebindCost(str, Enum):
    """What it costs to change a switch. Drives case ordering."""

    PER_REQUEST = "per_request"
    ENGINE_REBUILD = "engine_rebuild"
    PROCESS_GROUP_REBUILD = "process_group_rebuild"
    PROCESS_RESTART = "process_restart"


class SettingChannel(str, Enum):
    """How a setting reaches the engine, which decides when it takes effect."""

    ENV_VAR = "env_var"  # read once at process start -> PROCESS_RESTART
    TORCH_GLOBAL = "torch_global"  # torch.backends.* -> PROCESS_RESTART
    ENGINE_ARG = "engine_arg"  # engine constructor -> ENGINE_REBUILD
    CALL_ARG = "call_arg"  # passed per call -> PER_REQUEST


@dataclass(frozen=True)
class LibraryPin:
    """A pinned library version.

    TransformerEngine and FlashInfer change kernel selection across versions, so
    the same factor can reach the opposite conclusion on a different one. The pin
    goes into the execution fingerprint rather than a footnote.
    """

    package: str
    version: str  # exact; ranges are not accepted
    commit: str | None = None
    container_digest: str | None = None  # the only truly reproducible anchor


@dataclass(frozen=True)
class RequiredSetting:
    """A setting that must be pinned, and how to prove it was.

    A setting that cannot be read back can only be recorded ``UNOBSERVABLE``:
    delivered but unverifiable is the same as not delivered.
    """

    key: str
    value: Any
    channel: SettingChannel
    readback: str | None = None
    guards: str = ""  # KnownPitfall.id


@dataclass(frozen=True)
class PrecisionProfile:
    """Which precision this side uses at each point in the computation.

    Not ``PrecisionPolicy``: in RL, ``Policy`` is pi.
    """

    compute: Precision
    accumulate: Precision
    downcast_at: DowncastPoint
    master_weights: Precision | None = None
    lm_head: Precision | None = None  # should be FP32
    softmax_accumulate: Precision | None = None
    kv_accumulate: Precision | None = None  # linear attention KV, should be FP32


def choice_parser(*allowed: Any) -> Callable[[Any], Any]:
    """Build a parser accepting only the given values."""

    permitted = tuple(allowed)

    def parse(value: Any) -> Any:
        if value not in permitted:
            raise ValueError(f"expected one of {permitted!r}, got {value!r}")
        return value

    return parse


def positive_int(value: Any) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"expected a positive integer, got {value!r}")
    return parsed


def strict_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"expected a bool, got {value!r}")
    return value


__all__ = [
    "DowncastPoint",
    "ExecutionPath",
    "LibraryPin",
    "PolicyRole",
    "Precision",
    "PrecisionProfile",
    "RebindCost",
    "RequiredSetting",
    "SettingChannel",
    "choice_parser",
    "positive_int",
    "strict_bool",
]
