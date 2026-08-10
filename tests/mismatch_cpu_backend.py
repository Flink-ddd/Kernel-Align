# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""A CPU scoring backend for exercising the framework without a GPU.

Plumbing only: it says nothing about real Megatron or vLLM numerics. What it can
do is simulate the failure modes the framework exists to catch -- a per-side
bias, a switch that silently does nothing, and output that changes with the
environment -- so the gates and the matrix are tested against something that
really fails.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rl_engine.mismatch.schema import ComparisonIdentity, Evidence, PolicyRole, ReuseKey


@dataclass
class CpuScoringBackend:
    """Deterministic synthetic logprobs, with injectable deviation.

    Setting ``bias`` on one side only simulates a one-sided root cause.
    """

    role: PolicyRole
    bias: float = 0.0
    silently_ignores: frozenset[str] = frozenset()
    unstable_under: frozenset[str] = frozenset()
    evidence: frozenset[str] = frozenset(
        {
            Evidence.EFFECTIVE_CONFIG_READBACK.value,
            Evidence.MODEL_STATE_FINGERPRINT.value,
            Evidence.LIBRARY_VERSIONS.value,
        }
    )
    applied_calls: list[Mapping[str, Any]] = field(default_factory=list)

    def score(
        self,
        role: PolicyRole,
        identity: ComparisonIdentity,
        switch_values: Mapping[str, Any],
        replacement: Callable[..., Any] | None,
    ) -> tuple[Sequence[float], Mapping[str, Any]]:
        """Produce logprobs for the fixed sequence.

        The base signal depends only on the identity, so both sides agree unless
        a bias is injected.
        """

        self.applied_calls.append(dict(switch_values))
        base = _base_signal(identity)

        # The reference is defined to have no bias of its own.
        bias = 0.0 if replacement is not None else self.bias

        drift = 0.0
        for key in sorted(self.unstable_under):
            if key in switch_values:
                drift += _env_jitter(key, switch_values[key])

        logprobs = [value + bias + drift for value in base]

        effective = {
            key: ("ignored" if key in self.silently_ignores else value)
            for key, value in switch_values.items()
        }
        effective["evidence"] = tuple(self.evidence)
        return logprobs, effective

    def reuse_key(self, switch_values: Mapping[str, Any]) -> ReuseKey:
        """Group switches by tier, mirroring the four ``RebindCost`` levels."""

        def digest(prefix: str) -> str:
            payload = sorted(
                f"{key}={value}" for key, value in switch_values.items() if key.startswith(prefix)
            )
            return hashlib.sha256("|".join(payload).encode()).hexdigest()[:12]

        return ReuseKey(
            process=digest("env."),
            process_group=digest("dist."),
            engine=digest("engine."),
            request=digest("batch."),
        )


def _base_signal(identity: ComparisonIdentity) -> list[float]:
    """A stable pseudo-logprob per token, derived only from the identity."""

    tokens = identity.response_token_ids
    signal: list[float] = []
    for index, token in enumerate(tokens):
        seed = hashlib.sha256(f"{identity.checkpoint_id}:{index}:{token}".encode()).digest()
        magnitude = int.from_bytes(seed[:4], "big") / 0xFFFFFFFF
        signal.append(-(0.5 + magnitude))  # logprobs are negative
    return signal


def _env_jitter(key: str, value: Any) -> float:
    """Tiny environment-dependent shift, used to make instability observable."""

    seed = hashlib.sha256(f"{key}={value}".encode()).digest()
    return (int.from_bytes(seed[:2], "big") / 0xFFFF) * 1e-3


__all__ = ["CpuScoringBackend"]
