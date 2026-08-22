# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C5 (#271): on-chain elementwise / RoPE inventory.

C5 is a written audit with focused verdicts. Differentiable on-chain items
reuse C3/C4; items without a dedicated kernel are audited as pass-through
reductions. Kernel defects are Blockers, not silent N/A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Verdict = Literal["pass", "blocker", "blocked_hardware", "tracked_red", "absent_not_required"]


@dataclass(frozen=True)
class InventoryItem:
    name: str
    category: str
    on_chain: bool
    differentiable: bool
    entry_point: str
    reduction: str
    cuda_verdict: Verdict
    triton_verdict: Verdict
    evidence: str
    blocker: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "on_chain": self.on_chain,
            "differentiable": self.differentiable,
            "entry_point": self.entry_point,
            "reduction": self.reduction,
            "cuda_verdict": self.cuda_verdict,
            "triton_verdict": self.triton_verdict,
            "evidence": self.evidence,
            "blocker": self.blocker,
        }


# Blocker slugs until GitHub issues are filed from the #278 template.
BLOCKER_RMSNORM_DWEIGHT = "docs/design/ws1-blockers.md#rmsnorm-dweight"
BLOCKER_DET_GEMM_DW = "docs/design/ws1-blockers.md#det-gemm-dw"
BLOCKER_CUDA_LOGP_BWD = "docs/design/ws1-blockers.md#cuda-logp-no-backward"
BLOCKER_TRITON_ATTN_LEFT_PAD = "docs/design/ws1-blockers.md#triton-attention-left-pad"


ELEMENTWISE_INVENTORY: tuple[InventoryItem, ...] = (
    InventoryItem(
        name="rope",
        category="rope",
        on_chain=True,
        differentiable=True,
        entry_point="rl_engine.kernels.ops.{cuda,triton,pytorch}.rotary_embedding.rope",
        reduction="none (rotate_half, position-local)",
        cuda_verdict="pass",
        triton_verdict="pass",
        evidence=(
            "C3/C4 adapters registered; Triton green on sm86+; "
            "CUDA cuda-sm90 C3/C4 and C8 four-judgment green on H20"
        ),
    ),
    InventoryItem(
        name="silu",
        category="activation",
        on_chain=True,
        differentiable=True,
        entry_point="rl_engine.kernels.ops.{cuda,triton,pytorch}.activation.swiglu.SiLU*",
        reduction="none (pointwise)",
        cuda_verdict="pass",
        triton_verdict="pass",
        evidence="C3 and C4 both green on cuda_bf16 and triton_cuda_bf16 (sm86)",
    ),
    InventoryItem(
        name="swiglu",
        category="activation",
        on_chain=True,
        differentiable=True,
        entry_point="rl_engine.kernels.ops.{cuda,triton,pytorch}.activation.swiglu.SwiGLU*",
        reduction="none (pointwise gate*silu(up))",
        cuda_verdict="pass",
        triton_verdict="pass",
        evidence="C3 and C4 both green on cuda_bf16 and triton_cuda_bf16 (sm86)",
    ),
    InventoryItem(
        name="residual_add",
        category="residual",
        on_chain=True,
        differentiable=True,
        entry_point="torch.add (no dedicated WS1 kernel; C9 residual stream)",
        reduction="none (elementwise add, no cross-batch reduction)",
        cuda_verdict="pass",
        triton_verdict="pass",
        evidence=(
            "Audit: residual is x + y with matching logical tokens; "
            "no tile/batch-shape reduction. Covered by C3 token restore of surrounding ops"
        ),
    ),
    InventoryItem(
        name="scale",
        category="scale",
        on_chain=True,
        differentiable=True,
        entry_point="attention softmax scale = 1/sqrt(head_dim)",
        reduction="none (broadcast scalar)",
        cuda_verdict="pass",
        triton_verdict="pass",
        evidence="Pinned in Native/CUDA/Triton attention; independent of batch/layout",
    ),
    InventoryItem(
        name="bias",
        category="bias",
        on_chain=True,
        differentiable=False,
        entry_point="Qwen3-8B Dense: attention_bias=false; LM head bias=None",
        reduction="none (absent on the official fingerprint)",
        cuda_verdict="pass",
        triton_verdict="pass",
        evidence="C2 config_fingerprint.attention_bias is false; adapters pass bias=None",
    ),
    InventoryItem(
        name="mask_fill",
        category="mask",
        on_chain=True,
        differentiable=True,
        entry_point="attention key_padding_mask (True=keep)",
        reduction="none (masked fill to -inf before softmax)",
        cuda_verdict="pass",
        triton_verdict="pass",
        evidence=(
            "CUDA and Triton C3 padded_left are bitwise 0; Triton rebases the "
            "contiguous valid KV interval to logical reduction lanes"
        ),
    ),
    InventoryItem(
        name="dtype_cast",
        category="cast",
        on_chain=True,
        differentiable=False,
        entry_point="C1 dtype policy (BF16 exec, FP32 accumulate/reference)",
        reduction="none (policy cast, not a shape-dependent path)",
        cuda_verdict="pass",
        triton_verdict="pass",
        evidence="tolerance_contract.json policy; C3/C4 provenance rejects dtype drift",
    ),
)


def inventory_items() -> tuple[InventoryItem, ...]:
    return ELEMENTWISE_INVENTORY


def inventory_names() -> tuple[str, ...]:
    return tuple(item.name for item in ELEMENTWISE_INVENTORY)


def unresolved_needs_fix() -> tuple[InventoryItem, ...]:
    return tuple(
        item
        for item in ELEMENTWISE_INVENTORY
        if item.cuda_verdict == "blocker" or item.triton_verdict == "blocker"
    )


__all__ = [
    "BLOCKER_CUDA_LOGP_BWD",
    "BLOCKER_DET_GEMM_DW",
    "BLOCKER_RMSNORM_DWEIGHT",
    "BLOCKER_TRITON_ATTN_LEFT_PAD",
    "ELEMENTWISE_INVENTORY",
    "InventoryItem",
    "inventory_items",
    "inventory_names",
    "unresolved_needs_fix",
]
