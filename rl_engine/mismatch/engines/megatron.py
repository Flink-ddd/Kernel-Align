# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The training side: Megatron-LM. Not implemented.

Settings this backend must pin:

    AttnBackend                              explicit, never "auto"
    NVTE_ALLOW_NONDETERMINISTIC_ALGO         "0"
    CUBLAS_WORKSPACE_CONFIG                  ":4096:8"
    NCCL_ALGO / NCCL_PROTO                   pinned
    torch.backends.cuda.matmul.allow_tf32    False
    ...allow_bf16_reduced_precision_reduction  False
    torch.use_deterministic_algorithms       True
    lm_head dtype                            fp32

``AttnBackend=auto`` picks a different kernel per shape, so requested and actual
must be recorded separately. ``score()`` must leave the model untouched: a kernel
that mutates weights in place makes ``both_reference`` fail bitwise at random and
the blame lands on the reference. Under TP/CP no single rank holds the whole
logprob vector, so shards are collected per rank and counted against
``world_size``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rl_engine.mismatch.schema import ComparisonIdentity, LogprobShard, PolicyRole, ReuseKey


@dataclass
class MegatronBackend:
    role: PolicyRole = PolicyRole.TRAINING
    model: Any = None
    effective_config: dict[str, Any] = field(default_factory=dict)

    def score(
        self,
        role: PolicyRole,
        identity: ComparisonIdentity,
        switch_values: Mapping[str, Any],
        replacement: Callable[..., Any] | None,
    ) -> tuple[Sequence[float], Mapping[str, Any]]:
        raise NotImplementedError(
            "Run one forward over prompt + response, gather the selected "
            "logprobs, and assert the model state fingerprint is unchanged."
        )

    def reuse_key(self, switch_values: Mapping[str, Any]) -> ReuseKey:
        raise NotImplementedError(
            "Group switches by tier: determinism env -> process, TP/CP/PP -> "
            "process_group, dtype and kernel choice -> engine, batch -> request."
        )

    def read_effective_config(self) -> Mapping[str, Any]:
        raise NotImplementedError("Read each pinned setting off the live config.")

    def collect_logprob_shards(self) -> tuple[LogprobShard, ...]:
        raise NotImplementedError("Return one shard per rank.")


__all__ = ["MegatronBackend"]
