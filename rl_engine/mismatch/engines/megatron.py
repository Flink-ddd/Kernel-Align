# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The training side: Megatron-LM. **Placeholder -- not wired up yet.**

What has to be built here:

1. **Construction.** Model parallel state (TP/CP/PP) is fixed when the process
   group is built, so anything touching it costs
   ``RebindCost.PROCESS_GROUP_REBUILD``; determinism flags are read once at
   process start, so they cost ``PROCESS_RESTART``.
2. **Scoring.** ``score()`` returns per-token logprobs for a fixed sequence via a
   single forward over ``prompt + response`` -- ``ExecutionPath.TRAINING_FULL_PREFILL``,
   the training side's only path. It must be **read-only**: fingerprint the model
   tensors before and after and assert they match
   (``pipeline/runner.py::assert_comparison_is_read_only``). A kernel that
   mutates weights in place makes ``both_reference`` fail bitwise at random, and
   the blame lands on the reference.
3. **Readback.** Off the live config objects, never from what was requested.

Settings to pin and read back, with the pitfall each guards:

    AttnBackend                 -> explicit, never "auto"   guards requested_not_actual
      Megatron's auto picks a different kernel per shape, so requested and actual
      must be recorded separately (attn.provenance).

    NVTE_ALLOW_NONDETERMINISTIC_ALGO -> "0"                 os.environ
    CUBLAS_WORKSPACE_CONFIG          -> ":4096:8"           os.environ
    NCCL_ALGO / NCCL_PROTO           -> pinned              guards nccl_algo_unpinned

    torch.backends.cuda.matmul.allow_tf32                        -> False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction -> False
    torch.use_deterministic_algorithms(True)

    sequence_parallel           -> record
      With SP on, Megatron rewrites all_reduce into reduce_scatter + all_gather.
      Whether each side applies that rewrite is gemm.forward_reduce.

    lm_head dtype               -> fp32
      A BF16 head that is never upcast costs real logprob accuracy
      (logp.precision_downcast).

Under TP/CP no single rank holds the whole logprob vector: collect one
``LogprobShard`` per rank and let gate 3 check the count against ``world_size``.
A missing shard is wrong in a way that does not show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rl_engine.mismatch.schema import ComparisonIdentity, LogprobShard, PolicyRole, ReuseKey


@dataclass
class MegatronBackend:
    """Scores tokens on a live Megatron model. **Every method is a stub.**"""

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
            "Megatron scoring is not wired up: run one forward over "
            "prompt + response, gather the selected logprobs, and assert the "
            "model state fingerprint is unchanged across the call."
        )

    def reuse_key(self, switch_values: Mapping[str, Any]) -> ReuseKey:
        raise NotImplementedError(
            "Group the switches by tier: determinism env -> process, TP/CP/PP -> "
            "process_group, dtype and kernel choice -> engine, batch -> request."
        )

    def read_effective_config(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            "Read each pinned setting back off the live config. Requested is not "
            "actual, and Megatron's AttnBackend=auto is exactly where that bites."
        )

    def collect_logprob_shards(self) -> tuple[LogprobShard, ...]:
        raise NotImplementedError(
            "Under TP/CP each rank holds one slice; return one shard per rank so "
            "gate 3 can check the count against world_size."
        )


__all__ = ["MegatronBackend"]
