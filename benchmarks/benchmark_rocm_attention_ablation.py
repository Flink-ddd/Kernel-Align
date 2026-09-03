# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Execute the PR230 Attention row taxonomy as ROCm operator micro-probes.

This does not claim PR230's frozen model/rollout replay: there is no checkpoint,
token stream, selected-token logprob, KL, or serving engine in this benchmark.
Most rows use the shared native HIP deterministic Attention reference core; A6
and A7 use eager PyTorch-on-ROCm probes to isolate precision and merge order.
Every row records both implementations and the result scope explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from rl_engine.alignment.cross_config.attention_binding import (
    BindingErrorCode,
    bind_attention_contracts,
)
from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_REFERENCE_CORE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    AttentionContract,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    ReductionSpec,
    ShardingSpec,
    SplitKVSpec,
    build_split_kv_runtime_plan_set,
)
from rl_engine.kernels.ops.pytorch.attention.debug_matrix import attention_debug_matrix

QWEN3_Q_HEADS = 32
QWEN3_KV_HEADS = 8
QWEN3_HEAD_DIM = 128
DEFAULT_SHAPES = ((1, 16), (1, 32), (1, 64), (1, 128), (2, 16), (2, 32), (2, 64), (2, 128))
METRIC_NAMES = ("out", "lse", "dq", "dk", "dv")
ROCM_REFERENCE_BACKEND_ID = "rlkernel.rocm.deterministic_attention"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA = "rlkernel.rocm.attention_ablation_microprobe.v1"
RESULT_SCOPE = {
    "kind": "operator_microprobe",
    "pr230_row_taxonomy": True,
    "frozen_rollout_replay": False,
    "model_or_serving_execution": False,
    "covered_metrics": ["out_max_abs", "lse_max_abs", "dq_max_abs", "dk_max_abs", "dv_max_abs"],
    "excluded_metrics": [
        "train_rollout_logprob_abs_diff",
        "mismatch_kl",
        "mismatch_k3_kl",
    ],
}
ROW_IMPLEMENTATIONS = {
    "A0": (ROCM_REFERENCE_BACKEND_ID, ROCM_REFERENCE_BACKEND_ID),
    "A1": (
        "rlkernel.rocm.deterministic_rope+native_attention",
        "rlkernel.rocm.deterministic_rope+native_attention",
    ),
    "A2": ("torch.rocm.rms_norm+native_attention", ROCM_REFERENCE_BACKEND_ID),
    "A3": (ROCM_REFERENCE_BACKEND_ID, ROCM_REFERENCE_BACKEND_ID),
    "A5": (ROCM_REFERENCE_BACKEND_ID, "torch.rocm.index_select+native_attention"),
    "A6": (
        "torch.rocm.explicit_fp32_serial_qk_accumulator",
        "torch.rocm.explicit_bf16_serial_qk_accumulator",
    ),
    "A7": (
        "torch.rocm.fp32_chunk_merge_ascending",
        "torch.rocm.fp32_chunk_merge_descending",
    ),
    "C0": (ROCM_REFERENCE_BACKEND_ID, "rlkernel.rocm.native_attention_per_tp2_head_shard"),
    "C1": (ROCM_REFERENCE_BACKEND_ID, "rlkernel.rocm.native_attention_per_batch_row"),
    "C2": ("rlkernel.rocm.native_attention_full_prefill", "rlkernel.rocm.native_attention_tail"),
}
ROW_REALIZATIONS = {
    "A0": "Repeat the identical native HIP reference-core call.",
    "A1": "Increment suffix RoPE positions while preserving Q/K/V tensors.",
    "A2": "Apply or bypass unit-weight PyTorch RMSNorm before the native HIP core.",
    "A3": "Toggle causal masking in the native HIP core.",
    "A4": "Bind valid TP-rank-1 rollout and TP-rank-0 training contracts; do not run numerics.",
    "A5": "Reverse four dense K/V tensor pages; this is not a paged-cache runtime.",
    "A6": "Use identical FP32 products/order with explicit FP32 versus BF16 accumulator state.",
    "A7": "Merge four dense chunks in opposite orders on one GPU; this is not a CP collective.",
    "C0": "Compare full GQA with two contiguous TP=2 head shards on one GPU.",
    "C1": "Compare a batch call with per-row calls to the same native HIP core.",
    "C2": (
        "Compare a full-prefill tail with one trailing query over dense KV, without a serving "
        "cache."
    ),
}


@dataclass(frozen=True)
class Snapshot:
    out: torch.Tensor
    lse: torch.Tensor
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


AttentionCall = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def _seeded_tensor(shape: tuple[int, ...], *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)


def _evaluate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    call: AttentionCall,
) -> Snapshot:
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    out, lse = call(qr, kr, vr)
    if out.shape != dout.shape:
        raise RuntimeError(f"upstream gradient shape {dout.shape} does not match {out.shape}")
    dq, dk, dv = torch.autograd.grad(
        out,
        (qr, kr, vr),
        grad_outputs=dout,
        allow_unused=False,
    )
    return Snapshot(
        out.detach(),
        lse.detach(),
        dq.detach(),
        dk.detach(),
        dv.detach(),
    )


def _metric(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"metric tensors differ in shape: {left.shape} != {right.shape}")
    difference = (left.float() - right.float()).abs()
    same_dtype = left.dtype == right.dtype
    if same_dtype:
        left_bytes = left.contiguous().view(torch.uint8).reshape(left.numel(), left.element_size())
        right_bytes = (
            right.contiguous().view(torch.uint8).reshape(right.numel(), right.element_size())
        )
        mismatch_count = int(torch.any(left_bytes != right_bytes, dim=1).sum().item())
    else:
        mismatch_count = left.numel()
    return {
        "max_abs": 0.0 if difference.numel() == 0 else float(difference.max().item()),
        "mismatch_count": mismatch_count,
        "element_count": int(left.numel()),
        "bitwise_equal": mismatch_count == 0,
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "shape": list(left.shape),
    }


def _compare(left: Snapshot, right: Snapshot) -> dict[str, dict[str, Any]]:
    return {name: _metric(getattr(left, name), getattr(right, name)) for name in METRIC_NAMES}


def _native_call(operator: Any, *, causal: bool = True) -> AttentionCall:
    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        key_positions = torch.arange(k.size(2), device=k.device, dtype=torch.int64).repeat(
            k.size(0), 1
        )
        result = operator.forward_with_lse(
            q,
            k,
            v,
            causal=causal,
            scale=1.0 / math.sqrt(q.size(-1)),
            query_position_ids=key_positions[:, -q.size(2) :],
            key_position_ids=key_positions,
        )
        return result.out, result.lse

    return call


def _rope_batch(operator: Any, tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [operator(tensor[index : index + 1], positions[index]) for index in range(tensor.size(0))]
    )


def _rope_call(attention: Any, rope: Any, positions: torch.Tensor) -> AttentionCall:
    native = _native_call(attention)

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return native(_rope_batch(rope, q, positions), _rope_batch(rope, k, positions), v)

    return call


def _qk_norm_call(attention: Any, weight: torch.Tensor, *, enabled: bool) -> AttentionCall:
    native = _native_call(attention)

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if enabled:
            q = F.rms_norm(q, (q.size(-1),), weight, 1.0e-6)
            k = F.rms_norm(k, (k.size(-1),), weight, 1.0e-6)
        return native(q, k, v)

    return call


def _kv_page_call(attention: Any, permutation: torch.Tensor | None) -> AttentionCall:
    native = _native_call(attention)

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if permutation is not None:
            k = k.index_select(2, permutation)
            v = v.index_select(2, permutation)
        return native(q, k, v)

    return call


def _dense_attention(*, accumulator_dtype: torch.dtype) -> AttentionCall:
    if accumulator_dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("accumulator_dtype must be torch.float32 or torch.bfloat16")

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        group = q.size(1) // k.size(1)
        expanded_k = k.repeat_interleave(group, dim=1)
        expanded_v = v.repeat_interleave(group, dim=1)
        scores = torch.zeros(
            (*q.shape[:-1], expanded_k.size(2)),
            device=q.device,
            dtype=accumulator_dtype,
        )
        for index in range(q.size(-1)):
            product = q[..., index].float().unsqueeze(-1) * expanded_k[
                ..., index
            ].float().unsqueeze(-2)
            scores = scores.float() + product
            if accumulator_dtype is torch.bfloat16:
                scores = scores.to(torch.bfloat16)
        scores = scores.float()
        scores = scores * (1.0 / math.sqrt(q.size(-1)))
        q_index = torch.arange(q.size(2), device=q.device).unsqueeze(1)
        k_index = torch.arange(k.size(2), device=q.device).unsqueeze(0)
        scores = scores.masked_fill(k_index > q_index, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        out = torch.matmul(torch.softmax(scores, dim=-1), expanded_v.float()).to(q.dtype)
        return out, lse

    return call


def _chunked_attention(order: str) -> AttentionCall:
    if order not in {"ascending", "descending"}:
        raise ValueError("chunk merge order must be ascending or descending")

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        group = q.size(1) // k.size(1)
        expanded_k = k.repeat_interleave(group, dim=1)
        expanded_v = v.repeat_interleave(group, dim=1).float()
        sequence = k.size(2)
        chunk_size = sequence // 4
        chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        q_positions = torch.arange(q.size(2), device=q.device).view(1, 1, -1, 1)
        for start in range(0, sequence, chunk_size):
            stop = min(start + chunk_size, sequence)
            scores = torch.matmul(q.float(), expanded_k[:, :, start:stop].float().transpose(-1, -2))
            scores = scores * (1.0 / math.sqrt(q.size(-1)))
            key_positions = torch.arange(start, stop, device=q.device).view(1, 1, 1, -1)
            scores = scores.masked_fill(key_positions > q_positions, float("-inf"))
            row_max = scores.max(dim=-1).values
            valid = torch.isfinite(row_max)
            safe_max = torch.where(valid, row_max, torch.zeros_like(row_max))
            weights = torch.exp(scores - safe_max.unsqueeze(-1))
            weights = torch.where(valid.unsqueeze(-1), weights, torch.zeros_like(weights))
            denominator = weights.sum(dim=-1)
            numerator = torch.matmul(weights, expanded_v[:, :, start:stop])
            chunks.append((safe_max, denominator, numerator))

        indices = range(len(chunks)) if order == "ascending" else reversed(range(len(chunks)))
        merged_max: torch.Tensor | None = None
        merged_denominator: torch.Tensor | None = None
        merged_numerator: torch.Tensor | None = None
        for index in indices:
            row_max, denominator, numerator = chunks[index]
            if merged_max is None:
                merged_max, merged_denominator, merged_numerator = (
                    row_max,
                    denominator,
                    numerator,
                )
                continue
            assert merged_denominator is not None and merged_numerator is not None
            left_valid = merged_denominator > 0
            right_valid = denominator > 0
            new_max = torch.maximum(merged_max, row_max)
            left_scale = torch.where(
                left_valid,
                torch.exp(merged_max - new_max),
                torch.zeros_like(new_max),
            )
            right_scale = torch.where(
                right_valid,
                torch.exp(row_max - new_max),
                torch.zeros_like(new_max),
            )
            merged_denominator = merged_denominator * left_scale + denominator * right_scale
            merged_numerator = merged_numerator * left_scale.unsqueeze(
                -1
            ) + numerator * right_scale.unsqueeze(-1)
            merged_max = new_max
        assert merged_max is not None
        assert merged_denominator is not None and merged_numerator is not None
        out = (merged_numerator / merged_denominator.unsqueeze(-1)).to(q.dtype)
        lse = torch.log(merged_denominator) + merged_max
        return out, lse

    return call


def _per_tp_partition_call(attention: Any) -> AttentionCall:
    native = _native_call(attention)

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        tp_world_size = 2
        q_heads_per_rank = q.size(1) // tp_world_size
        kv_heads_per_rank = k.size(1) // tp_world_size
        outputs, lses = [], []
        for rank in range(tp_world_size):
            result = native(
                q[:, rank * q_heads_per_rank : (rank + 1) * q_heads_per_rank],
                k[:, rank * kv_heads_per_rank : (rank + 1) * kv_heads_per_rank],
                v[:, rank * kv_heads_per_rank : (rank + 1) * kv_heads_per_rank],
            )
            outputs.append(result[0])
            lses.append(result[1])
        return torch.cat(outputs, dim=1), torch.cat(lses, dim=1)

    return call


def _per_batch_row_call(attention: Any) -> AttentionCall:
    native = _native_call(attention)

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        rows = [native(q[i : i + 1], k[i : i + 1], v[i : i + 1]) for i in range(q.size(0))]
        return torch.cat([row[0] for row in rows]), torch.cat([row[1] for row in rows])

    return call


def _tail_call(attention: Any, *, full_query: bool) -> AttentionCall:
    native = _native_call(attention)

    def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if full_query:
            out, lse = native(q, k, v)
            return out[:, :, -1:], lse[:, :, -1:]
        return native(q[:, :, -1:], k, v)

    return call


def _page_permutation(sequence: int, device: torch.device) -> torch.Tensor:
    page_size = max(1, sequence // 4)
    pages = [
        torch.arange(start, min(start + page_size, sequence), device=device)
        for start in range(0, sequence, page_size)
    ]
    return torch.cat(list(reversed(pages))).to(torch.long)


def _topology_contract(
    *, role: AttentionRole, batch: int, sequence: int, tp_rank: int
) -> AttentionContract:
    tp_world_size = 2
    local_q_heads = QWEN3_Q_HEADS // tp_world_size
    local_kv_heads = QWEN3_KV_HEADS // tp_world_size
    return AttentionContract(
        role=role,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=batch,
        query_sequence_length=sequence,
        head_dim=QWEN3_HEAD_DIM,
        causal=True,
        causal_offsets=(0,) * batch,
        sharding=ShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            cp_rank=0,
            cp_world_size=1,
            global_q_heads=QWEN3_Q_HEADS,
            global_kv_heads=QWEN3_KV_HEADS,
            local_q_head_start=tp_rank * local_q_heads,
            local_q_heads=local_q_heads,
            local_kv_head_start=tp_rank * local_kv_heads,
            local_kv_heads=local_kv_heads,
            global_sequence_length=sequence,
            local_sequence_length=sequence,
            global_block_indices=(0,),
            global_block_token_starts=(0,),
            local_block_offsets=(0, sequence),
        ),
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
    )


def _topology_gate(*, batch: int, sequence: int) -> dict[str, Any]:
    """Exercise the production contract gate with different TP owners."""

    rollout = _topology_contract(
        role=AttentionRole.INFER,
        batch=batch,
        sequence=sequence,
        tp_rank=1,
    )
    training = _topology_contract(
        role=AttentionRole.TRAIN,
        batch=batch,
        sequence=sequence,
        tp_rank=0,
    )
    plan_set = build_split_kv_runtime_plan_set(
        (sequence,) * batch,
        tp_world_size=2,
        cp_world_size=1,
        split_kv=SplitKVSpec.disabled(),
        backend=ROCM_REFERENCE_BACKEND_ID,
    )
    result = bind_attention_contracts(
        rollout_contract=rollout,
        training_contract=training,
        rollout_identity={},
        training_identity={},
        rollout_backend_id=ROCM_REFERENCE_BACKEND_ID,
        training_backend_id=ROCM_REFERENCE_BACKEND_ID,
        rollout_split_kv_plan_set=plan_set,
        training_split_kv_plan_set=plan_set,
        require_full_identity=False,
    )
    topology_issues = result.issues_by_code(BindingErrorCode.TOPOLOGY_MISMATCH)
    unexpected_issues = tuple(
        issue for issue in result.issues if issue.code is not BindingErrorCode.TOPOLOGY_MISMATCH
    )
    if result.comparable or result.passed or not topology_issues or unexpected_issues:
        raise RuntimeError("A4 did not isolate the topology comparability gate")
    return result.to_dict()


def _case(
    *,
    batch: int,
    sequence: int,
    seed: int,
    device: torch.device,
    attention: Any,
    rope: Any,
) -> list[dict[str, Any]]:
    q = _seeded_tensor((batch, QWEN3_Q_HEADS, sequence, QWEN3_HEAD_DIM), device=device, seed=seed)
    k = _seeded_tensor(
        (batch, QWEN3_KV_HEADS, sequence, QWEN3_HEAD_DIM), device=device, seed=seed + 1
    )
    v = _seeded_tensor(k.shape, device=device, seed=seed + 2)
    dout = _seeded_tensor(q.shape, device=device, seed=seed + 3)
    native = _native_call(attention)

    positions = torch.arange(sequence, device=device, dtype=torch.int64).repeat(batch, 1)
    changed_positions = positions.clone()
    changed_positions[:, sequence // 2 :] += 1
    norm_weight = torch.ones(QWEN3_HEAD_DIM, device=device, dtype=torch.bfloat16)
    permutation = _page_permutation(sequence, device)

    calls: dict[str, tuple[AttentionCall, AttentionCall, torch.Tensor]] = {
        "A0": (native, native, dout),
        "A1": (
            _rope_call(attention, rope, positions),
            _rope_call(attention, rope, changed_positions),
            dout,
        ),
        "A2": (
            _qk_norm_call(attention, norm_weight, enabled=True),
            _qk_norm_call(attention, norm_weight, enabled=False),
            dout,
        ),
        "A3": (native, _native_call(attention, causal=False), dout),
        "A5": (native, _kv_page_call(attention, permutation), dout),
        "A6": (
            _dense_attention(accumulator_dtype=torch.float32),
            _dense_attention(accumulator_dtype=torch.bfloat16),
            dout,
        ),
        "A7": (_chunked_attention("ascending"), _chunked_attention("descending"), dout),
        "C0": (native, _per_tp_partition_call(attention), dout),
        "C1": (native, _per_batch_row_call(attention), dout),
        "C2": (
            _tail_call(attention, full_query=True),
            _tail_call(attention, full_query=False),
            dout[:, :, -1:].contiguous(),
        ),
    }
    rows: list[dict[str, Any]] = []
    for matrix_row in attention_debug_matrix()["rows"]:
        row_id = matrix_row["id"]
        if row_id == "A4":
            binding = _topology_gate(batch=batch, sequence=sequence)
            rows.append(
                {
                    "row_id": row_id,
                    "batch": batch,
                    "sequence": sequence,
                    "category": matrix_row["category"],
                    "probe": matrix_row["probe"],
                    "expected": matrix_row["expected"],
                    "comparable": binding["comparable"],
                    "passed": True,
                    "outcome": "rejected",
                    "realization": ROW_REALIZATIONS[row_id],
                    "gate_implementation": (
                        "rl_engine.alignment.cross_config.bind_attention_contracts"
                    ),
                    "identity_errors": [issue["field"] for issue in binding["issues"]],
                    "binding_gate": binding,
                    "metrics": {name: None for name in METRIC_NAMES},
                }
            )
            continue
        baseline_call, candidate_call, row_dout = calls[row_id]
        baseline = _evaluate(q, k, v, row_dout, baseline_call)
        candidate = _evaluate(q, k, v, row_dout, candidate_call)
        metrics = _compare(baseline, candidate)
        mismatch_count = sum(metric["mismatch_count"] for metric in metrics.values())
        expected = matrix_row["expected"]
        passed = (
            mismatch_count == 0 if expected in {"baseline", "exact_zero"} else mismatch_count > 0
        )
        baseline_implementation, candidate_implementation = ROW_IMPLEMENTATIONS[row_id]
        rows.append(
            {
                "row_id": row_id,
                "batch": batch,
                "sequence": sequence,
                "category": matrix_row["category"],
                "probe": matrix_row["probe"],
                "expected": expected,
                "comparable": True,
                "passed": passed,
                "outcome": "matched" if mismatch_count == 0 else "drift_detected",
                "realization": ROW_REALIZATIONS[row_id],
                "implementations": {
                    "baseline": baseline_implementation,
                    "candidate": candidate_implementation,
                },
                "metrics": metrics,
            }
        )
    return rows


def _environment(device: torch.device, attention: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "hip_runtime": torch.version.hip,
        "device_index": device.index,
        "device_name": properties.name,
        "architecture": getattr(properties, "gcnArchName", "unknown"),
        "gpu_count": torch.cuda.device_count(),
        "primary_backend_id": attention.backend_id,
        "primary_core_id": attention.core_id,
        "primary_schedule": attention.strict_schedule,
        "primary_reference_only": attention.reference_only,
        "primary_production_ready": attention.production_ready,
        "execution_kind": "operator_only_rocm_reference",
    }


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )


def _source_provenance() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    script_relative = script_path.relative_to(REPO_ROOT).as_posix()
    revision_result = _git("rev-parse", "HEAD")
    head_source = _git("show", f"HEAD:{script_relative}")
    diff = _git("diff", "--binary", "HEAD")
    if revision_result.returncode != 0 or diff.returncode != 0:
        raise RuntimeError("unable to record RL-Kernel git provenance")
    source = script_path.read_bytes()
    tracked_diff = diff.stdout
    return {
        "revision": revision_result.stdout.decode().strip(),
        "tracked_dirty": bool(tracked_diff),
        "tracked_diff_sha256": (hashlib.sha256(tracked_diff).hexdigest() if tracked_diff else None),
        "script_path": script_relative,
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "script_matches_head": head_source.returncode == 0 and head_source.stdout == source,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    for matrix_row in attention_debug_matrix()["rows"]:
        selected = [row for row in rows if row["row_id"] == matrix_row["id"]]
        metrics: dict[str, Any] = {}
        for name in METRIC_NAMES:
            values = [row["metrics"][name] for row in selected if row["metrics"][name] is not None]
            metrics[name] = (
                None
                if not values
                else {
                    "worst_max_abs": max(value["max_abs"] for value in values),
                    "total_mismatch_count": sum(value["mismatch_count"] for value in values),
                    "all_bitwise_equal": all(value["bitwise_equal"] for value in values),
                }
            )
        aggregate.append(
            {
                **matrix_row,
                "case_count": len(selected),
                "comparable": all(row["comparable"] for row in selected),
                "passed": all(row["passed"] for row in selected),
                "metrics": metrics,
            }
        )
    return aggregate


def _valid_digest(value: Any, *, lengths: tuple[int, ...]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_metric(
    row_id: str,
    name: str,
    metric: Any,
    *,
    expected_dtype: str,
    expected_shape: list[int],
) -> None:
    if not isinstance(metric, Mapping) or set(metric) != {
        "max_abs",
        "mismatch_count",
        "element_count",
        "bitwise_equal",
        "left_dtype",
        "right_dtype",
        "shape",
    }:
        raise ValueError(f"{row_id}.{name} has an invalid metric schema")
    maximum = metric["max_abs"]
    mismatches = metric["mismatch_count"]
    elements = metric["element_count"]
    bitwise_equal = metric["bitwise_equal"]
    shape = metric["shape"]
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(maximum)
        or maximum < 0
    ):
        raise ValueError(f"{row_id}.{name}.max_abs must be finite and non-negative")
    if (
        isinstance(mismatches, bool)
        or not isinstance(mismatches, int)
        or isinstance(elements, bool)
        or not isinstance(elements, int)
        or elements < 1
        or mismatches < 0
        or mismatches > elements
    ):
        raise ValueError(f"{row_id}.{name} has invalid element or mismatch counts")
    if not isinstance(bitwise_equal, bool) or bitwise_equal != (mismatches == 0):
        raise ValueError(f"{row_id}.{name} has inconsistent bitwise evidence")
    if mismatches == 0 and maximum != 0:
        raise ValueError(f"{row_id}.{name} has inconsistent numerical evidence")
    if (
        metric["left_dtype"] != metric["right_dtype"]
        or metric["left_dtype"] != expected_dtype
        or shape != expected_shape
        or elements != math.prod(expected_shape)
    ):
        raise ValueError(f"{row_id}.{name} has incompatible dtype or shape evidence")


def validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("unsupported ROCm Attention ablation result schema")
    if payload.get("scope") != RESULT_SCOPE:
        raise ValueError("result scope must identify an operator micro-probe, not a model replay")
    manifest = attention_debug_matrix()
    if payload.get("matrix_manifest") != manifest:
        raise ValueError("result does not embed the exact PR230 matrix manifest")

    environment = payload.get("environment", {})
    if not isinstance(environment, Mapping) or (
        not environment.get("hip_runtime")
        or environment.get("execution_kind") != "operator_only_rocm_reference"
        or environment.get("primary_backend_id") != ROCM_REFERENCE_BACKEND_ID
        or environment.get("primary_core_id") != STRICT_ATTENTION_REFERENCE_CORE_ID
        or environment.get("primary_schedule") != STRICT_ATTENTION_SCHEDULE_ID
        or environment.get("primary_reference_only") is not True
        or environment.get("primary_production_ready") is not False
        or "gfx942" not in environment.get("architecture", "")
    ):
        raise ValueError("result does not prove gfx942 ROCm reference execution")
    for name in ("python", "pytorch", "hip_runtime", "device_name", "architecture"):
        if not isinstance(environment.get(name), str) or not environment[name].strip():
            raise ValueError(f"environment.{name} must be a non-empty runtime readback")
    hip_parts = environment["hip_runtime"].split(".")
    if (
        len(hip_parts) < 2
        or not hip_parts[0].isdigit()
        or not hip_parts[1].isdigit()
        or "rocm" not in environment["pytorch"].lower()
        or "amd" not in environment["device_name"].lower()
        or "mi300x" not in environment["device_name"].lower()
    ):
        raise ValueError("environment does not identify an AMD MI300X ROCm runtime")
    device_index = environment.get("device_index")
    gpu_count = environment.get("gpu_count")
    if (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or isinstance(gpu_count, bool)
        or not isinstance(gpu_count, int)
        or device_index < 0
        or gpu_count < 1
        or device_index >= gpu_count
    ):
        raise ValueError("environment GPU selection is invalid")

    command = payload.get("command")
    if (
        not isinstance(command, list)
        or len(command) < 2
        or any(not isinstance(argument, str) for argument in command)
        or not command[1].endswith("benchmark_rocm_attention_ablation.py")
    ):
        raise ValueError("result does not record the benchmark command")

    provenance = payload.get("source_provenance", {})
    if not isinstance(provenance, Mapping) or (
        not _valid_digest(provenance.get("revision"), lengths=(40, 64))
        or provenance.get("tracked_dirty") is not False
        or provenance.get("tracked_diff_sha256") is not None
        or provenance.get("script_path") != "benchmarks/benchmark_rocm_attention_ablation.py"
        or not _valid_digest(provenance.get("script_sha256"), lengths=(64,))
        or provenance.get("script_matches_head") is not True
    ):
        raise ValueError("result is not pinned to a clean committed runner")

    configuration = payload.get("configuration", {})
    raw_shapes = configuration.get("shapes") if isinstance(configuration, Mapping) else None
    if not isinstance(raw_shapes, list) or any(
        not isinstance(shape, (list, tuple)) or len(shape) != 2 for shape in raw_shapes
    ):
        raise ValueError("configuration.shapes must contain BxS pairs")
    try:
        shapes = tuple((shape[0], shape[1]) for shape in raw_shapes)
    except (IndexError, TypeError):
        raise ValueError("configuration.shapes must contain BxS pairs") from None
    if (
        not shapes
        or len(set(shapes)) != len(shapes)
        or any(
            isinstance(batch, bool)
            or isinstance(sequence, bool)
            or not isinstance(batch, int)
            or not isinstance(sequence, int)
            or batch < 1
            or sequence < 4
            or sequence % 4
            for batch, sequence in shapes
        )
    ):
        raise ValueError("configuration.shapes contains an invalid or duplicate shape")
    if shapes != DEFAULT_SHAPES:
        raise ValueError("publication results must cover the exact eight-shape ROCm sweep")
    expected_configuration = {
        "dtype": "bfloat16",
        "q_heads": QWEN3_Q_HEADS,
        "kv_heads": QWEN3_KV_HEADS,
        "head_dim": QWEN3_HEAD_DIM,
    }
    if any(configuration.get(name) != value for name, value in expected_configuration.items()):
        raise ValueError("result does not use the pinned Qwen3 BF16 Attention configuration")
    seed = configuration.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("configuration.seed must be a non-negative integer")

    expected_rows = [row["id"] for row in manifest["rows"]]
    cases = payload.get("cases", [])
    expected_case_keys = [
        (row_id, batch, sequence) for batch, sequence in shapes for row_id in expected_rows
    ]
    if (
        not isinstance(cases, list)
        or [
            (case.get("row_id"), case.get("batch"), case.get("sequence"))
            for case in cases
            if isinstance(case, Mapping)
        ]
        != expected_case_keys
    ):
        raise ValueError("cases do not cover every PR230 row and configured shape exactly once")

    manifest_by_id = {row["id"]: row for row in manifest["rows"]}
    for case in cases:
        row_id = case["row_id"]
        row_manifest = manifest_by_id[row_id]
        if any(case.get(name) != row_manifest[name] for name in ("category", "probe", "expected")):
            raise ValueError(f"{row_id} case metadata differs from the PR230 manifest")
        if case.get("realization") != ROW_REALIZATIONS[row_id]:
            raise ValueError(f"{row_id} does not disclose its operator-level realization")
        if case.get("passed") is not True:
            raise ValueError(f"{row_id} case did not satisfy its expected outcome")
        metrics = case.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
            raise ValueError(f"{row_id} has an invalid metric set")
        if row_id == "A4":
            binding = case.get("binding_gate", {})
            expected_binding = _topology_gate(batch=case["batch"], sequence=case["sequence"])
            issues = binding.get("issues", []) if isinstance(binding, Mapping) else []
            issue_evidence = [
                (
                    issue.get("code"),
                    issue.get("tier"),
                    issue.get("field"),
                    issue.get("rollout"),
                    issue.get("training"),
                )
                for issue in issues
                if isinstance(issue, Mapping)
            ]
            expected_issue_evidence = [
                ("TOPOLOGY_MISMATCH", "identical", "sharding.tp_rank", 1, 0),
                (
                    "TOPOLOGY_MISMATCH",
                    "identical",
                    "sharding.local_q_head_start",
                    QWEN3_Q_HEADS // 2,
                    0,
                ),
                (
                    "TOPOLOGY_MISMATCH",
                    "identical",
                    "sharding.local_kv_head_start",
                    QWEN3_KV_HEADS // 2,
                    0,
                ),
            ]
            if (
                case.get("comparable") is not False
                or case.get("outcome") != "rejected"
                or case.get("gate_implementation")
                != "rl_engine.alignment.cross_config.bind_attention_contracts"
                or any(metrics[name] is not None for name in METRIC_NAMES)
                or binding != expected_binding
                or binding.get("comparable") is not False
                or binding.get("passed") is not False
                or binding.get("schema_version") != "cross_config.attention_binding.v3"
                or not issues
                or any(not isinstance(issue, Mapping) for issue in issues)
                or issue_evidence != expected_issue_evidence
                or case.get("identity_errors") != [issue.get("field") for issue in issues]
            ):
                raise ValueError("A4 lacks isolated topology-gate rejection evidence")
            continue

        if case.get("comparable") is not True:
            raise ValueError(f"{row_id} must contain a numerical comparison")
        implementations = case.get("implementations", {})
        expected_implementations = ROW_IMPLEMENTATIONS[row_id]
        if implementations != {
            "baseline": expected_implementations[0],
            "candidate": expected_implementations[1],
        }:
            raise ValueError(f"{row_id} implementation provenance is missing or incorrect")
        batch, sequence = case["batch"], case["sequence"]
        expected_shapes = {
            "out": [batch, QWEN3_Q_HEADS, 1 if row_id == "C2" else sequence, QWEN3_HEAD_DIM],
            "lse": [batch, QWEN3_Q_HEADS, 1 if row_id == "C2" else sequence],
            "dq": [batch, QWEN3_Q_HEADS, sequence, QWEN3_HEAD_DIM],
            "dk": [batch, QWEN3_KV_HEADS, sequence, QWEN3_HEAD_DIM],
            "dv": [batch, QWEN3_KV_HEADS, sequence, QWEN3_HEAD_DIM],
        }
        for name in METRIC_NAMES:
            _validate_metric(
                row_id,
                name,
                metrics[name],
                expected_dtype="torch.float32" if name == "lse" else "torch.bfloat16",
                expected_shape=expected_shapes[name],
            )
        mismatch_count = sum(metrics[name]["mismatch_count"] for name in METRIC_NAMES)
        if row_manifest["expected"] in {"baseline", "exact_zero"}:
            valid_outcome = mismatch_count == 0 and case.get("outcome") == "matched"
        else:
            valid_outcome = mismatch_count > 0 and case.get("outcome") == "drift_detected"
        if not valid_outcome:
            raise ValueError(f"{row_id} numerical evidence contradicts its expected outcome")

    aggregates = payload.get("matrix", [])
    if not isinstance(aggregates, list) or any(not isinstance(row, Mapping) for row in aggregates):
        raise ValueError("matrix summary must be a list of rows")
    if [row.get("id") for row in aggregates] != expected_rows:
        raise ValueError("result rows do not match the PR230 matrix")
    reproduced = _aggregate(cases)
    for row in reproduced:
        if row["expected"] == "diagnostic" and any(
            row["metrics"][name]["total_mismatch_count"] == 0 for name in METRIC_NAMES
        ):
            raise ValueError(f"{row['id']} must show a drift signature in all five metrics")
    if aggregates != reproduced:
        raise ValueError("matrix summary does not reproduce the per-shape case evidence")
    failed = [row["id"] for row in aggregates if row.get("passed") is not True]
    if failed:
        raise ValueError("ROCm Attention ablation expectations failed: " + ", ".join(failed))


def validate_repository_provenance(payload: Mapping[str, Any]) -> None:
    """Bind recorded hashes to a real git object and the current runner."""

    provenance = payload["source_provenance"]
    revision = provenance["revision"]
    script_path = provenance["script_path"]
    committed_source = _git("show", f"{revision}:{script_path}")
    current_source = (REPO_ROOT / script_path).read_bytes()
    if (
        committed_source.returncode != 0
        or hashlib.sha256(committed_source.stdout).hexdigest() != provenance["script_sha256"]
        or hashlib.sha256(current_source).hexdigest() != provenance["script_sha256"]
    ):
        raise ValueError("recorded runner hash is not backed by the RL-Kernel repository")


def _format_metric(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "—"
    if value["total_mismatch_count"] == 0:
        return "`0`"
    return f"`{value['worst_max_abs']:.8g}`"


def _write_report(payload: Mapping[str, Any], path: Path) -> None:
    environment = payload["environment"]
    shapes = ", ".join(
        f"B={batch}, S={sequence}" for batch, sequence in payload["configuration"]["shapes"]
    )
    lines = [
        "# PR230 Attention taxonomy: ROCm operator micro-probes",
        "",
        "> This applies PR230's row taxonomy to deterministic operator micro-probes.",
        "> It is not the frozen model/rollout replay from PR230: no checkpoint, token stream,",
        "> selected-token logprob, KL, serving engine, or AITER production claim is included.",
        "",
        "## Environment",
        "",
        f"- GPU: {environment['device_name']} ({environment['architecture']})",
        f"- PyTorch: {environment['pytorch']}; HIP: {environment['hip_runtime']}",
        f"- RL-Kernel: `{payload['source_provenance']['revision']}`",
        f"- Shapes: {shapes}",
        "- Primary core: `rlkernel.rocm.deterministic_attention`",
        "",
        "## Matrix",
        "",
        "| Row | Factor | Comparable | Out | LSE | dQ | dK | dV | Result |",
        "|---|---|:---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["matrix"]:
        metrics = row["metrics"]
        values = " | ".join(_format_metric(metrics[name]) for name in METRIC_NAMES)
        result = "REJECTED" if row["id"] == "A4" and row["passed"] else "PASS"
        if not row["passed"]:
            result = "FAIL"
        lines.append(
            f"| {row['id']} | {row['label']} | {'yes' if row['comparable'] else 'no'} | "
            f"{values} | **{result}** |"
        )
    lines.extend(["", "## Probe realizations", ""])
    for row in attention_debug_matrix()["rows"]:
        lines.append(f"- `{row['id']}` — {ROW_REALIZATIONS[row['id']]}")
    lines.extend(
        [
            "",
            "A1-A3 and A5-A7 deliberately inject one mismatch and report the worst max-absolute",
            "difference over the shape sweep. A4 is rejected by the repository's cross-config",
            "binding gate before numerical comparison. A0 and C0-C2 must be bitwise zero for",
            "Out/LSE/dQ/dK/dV.",
            "",
            "A6 and A7 are eager PyTorch-on-ROCm probes for accumulation and merge order; the",
            "remaining numerical rows invoke the native deterministic HIP Attention core. This",
            "is operator-only reference evidence, not full PR230 replay evidence.",
            "",
            "The complete per-shape mismatch counts and max-absolute values are in `results.json`.",
            "",
            "## Reproduce",
            "",
            "Run from the recorded clean commit and choose a new output directory:",
            "",
            "```bash",
            "HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python \\",
            "  benchmarks/benchmark_rocm_attention_ablation.py --device 0 \\",
            "  --output-dir /tmp/pr230_rocm_mi300x_ablation",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/pr230_rocm_mi300x_ablation"),
    )
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.output_dir}")
    source_provenance = _source_provenance()
    if source_provenance["tracked_dirty"] or not source_provenance["script_matches_head"]:
        raise SystemExit("refusing to publish evidence from an uncommitted runner or tracked tree")
    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit("the ROCm Attention ablation matrix requires a ROCm GPU")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    if "gfx942" not in architecture:
        raise SystemExit(f"the checked-in matrix is pinned to gfx942, got {architecture!r}")

    from rl_engine.kernels.ops.cuda.attention.deterministic_attn import (
        RLKernelDeterministicAttentionCore,
    )
    from rl_engine.kernels.ops.cuda.rotary_embedding.rope import RocmDeterministicRoPEOp

    attention = RLKernelDeterministicAttentionCore()
    if attention.backend_id != ROCM_REFERENCE_BACKEND_ID:
        raise SystemExit(f"unexpected reference backend on ROCm: {attention.backend_id}")
    rope = RocmDeterministicRoPEOp()
    rows: list[dict[str, Any]] = []
    for index, (batch, sequence) in enumerate(DEFAULT_SHAPES):
        rows.extend(
            _case(
                batch=batch,
                sequence=sequence,
                seed=args.seed + index * 10,
                device=device,
                attention=attention,
                rope=rope,
            )
        )
        torch.cuda.synchronize(device)

    payload = {
        "schema_version": RESULT_SCHEMA,
        "scope": RESULT_SCOPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "source_provenance": source_provenance,
        "matrix_manifest": attention_debug_matrix(),
        "environment": _environment(device, attention),
        "configuration": {
            "seed": args.seed,
            "dtype": "bfloat16",
            "q_heads": QWEN3_Q_HEADS,
            "kv_heads": QWEN3_KV_HEADS,
            "head_dim": QWEN3_HEAD_DIM,
            "shapes": [list(shape) for shape in DEFAULT_SHAPES],
        },
        "cases": rows,
        "matrix": _aggregate(rows),
    }
    validate_payload(payload)
    validate_repository_provenance(payload)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(payload, args.output_dir / "report.md")
    print(json.dumps({"output_dir": str(args.output_dir), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
