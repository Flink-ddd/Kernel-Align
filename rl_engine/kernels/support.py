# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework-consumable backend support descriptors."""

from __future__ import annotations

from typing import Final

_LINEAR_LOGP_SUPPORT_MATRIX: Final[tuple[dict[str, str], ...]] = (
    {
        "source": "rl_kernel",
        "backend": "cuda_sm90",
        "implementation": "FusedLinearLogpSM90Op selected through rl_engine.kernels.registry",
        "dtype": "backend-selected floating contract; optimized paths target bf16/fp32 logprob use",
        "hardware": "NVIDIA SM90/Hopper CUDA when the compiled extension is available",
        "tp": (
            "supports vocab-parallel calls when tp_group, vocab_start_index, "
            "and global_vocab_size describe the local shard"
        ),
        "cp": (
            "not a context-parallel redistribution primitive; frameworks decide "
            "CP materialization or fallback"
        ),
        "entropy": (
            "not produced by linear_logp; frameworks that need entropy compute it separately"
        ),
        "full_gradient": (
            "supported when the selected path preserves backward state for hidden, weight, and bias"
        ),
    },
    {
        "source": "rl_kernel",
        "backend": "triton",
        "implementation": "TritonLinearLogpOp selected through rl_engine.kernels.registry",
        "dtype": "backend-selected floating input/output contract",
        "hardware": "CUDA devices supported by the installed Triton backend",
        "tp": (
            "supports vocab-parallel calls when the selected implementation accepts shard metadata"
        ),
        "cp": (
            "not a context-parallel redistribution primitive; frameworks decide "
            "CP materialization or fallback"
        ),
        "entropy": (
            "not produced by linear_logp; frameworks that need entropy compute it separately"
        ),
        "full_gradient": "backend-defined; strict callers should verify autograd-connected outputs",
    },
    {
        "source": "rl_kernel",
        "backend": "pytorch",
        "implementation": "NativeLinearLogpOp selected through rl_engine.kernels.registry",
        "dtype": "floating input/output contract implemented with PyTorch reference math",
        "hardware": "CPU, CUDA, or other PyTorch-supported tensor devices",
        "tp": "supports tensor-parallel semantics through the shared PyTorch linear_logp helpers",
        "cp": (
            "not a context-parallel redistribution primitive; frameworks decide "
            "CP materialization or fallback"
        ),
        "entropy": (
            "not produced by linear_logp; frameworks that need entropy compute it separately"
        ),
        "full_gradient": "supported through PyTorch autograd",
    },
)


def get_linear_logp_support_matrix() -> tuple[dict[str, str], ...]:
    """Return RL-Kernel-owned linear_logp backend support descriptors."""

    return tuple(dict(row) for row in _LINEAR_LOGP_SUPPORT_MATRIX)


__all__ = ["get_linear_logp_support_matrix"]
