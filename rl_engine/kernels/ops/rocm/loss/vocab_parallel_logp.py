"""ROCm WS2 vocab-parallel logprob backend.

The reference operator owns the contract: TP transport, fixed global tile-order
merge, target ownership, masking, entropy, and autograd semantics.  This backend
keeps all of that and replaces the two large per-shard passes with HIP kernels
from ``csrc/hip/hip_deterministic_logp_kernel.hip`` (compiled only for ROCm; the
shared ``csrc/deterministic_logp_kernel.cu`` keeps the SM90-tuned CUDA path):

* ``hip_deterministic_logp_tile_stats`` computes the per-row, per-tile FP32
  ``(max, sumexp)`` partials straight from the BF16/FP16/FP32 shard.  The kernel
  converts each element exactly, filters padding columns itself, and reduces
  every tile with a fixed tree, so no FP32 copy of the logits is materialized.
* ``hip_deterministic_logp_backward`` produces ``grad_logits`` for the selected
  logprob and LSE outputs in one fused pass from the saved input shard.

``apply`` runs through the shared :func:`apply_with_kernels` path;
``apply_with_entropy`` keeps the shared autograd path (with the HIP tile
kernel) because the entropy gradient needs the full probability tensor anyway.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.kernels.logprob_contract import LogprobContract
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import (
    DEFAULT_NUM_VOCAB_TILES,
    VocabParallelLogprobOp,
    apply_with_kernels,
)

BACKEND_ID = "rocm-vocab-parallel-logp-ws2"


def _native_backward_available() -> bool:
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    except ImportError:
        return False
    return bool(
        _EXT_AVAILABLE
        and hasattr(_C, "hip_deterministic_logp_tile_stats")
        and hasattr(_C, "hip_deterministic_logp_backward")
    )


class _HipKernels:
    """``VocabParallelLogprobKernels`` over the ROCm extension symbols."""

    @staticmethod
    def tile_stats(shard, vocab_start, real_vocab, num_tiles):
        from rl_engine.kernels.ops.base import _C

        tile_max, tile_sum = _C.hip_deterministic_logp_tile_stats(
            shard, vocab_start, real_vocab, num_tiles
        )
        return tile_max, tile_sum

    @staticmethod
    def backward(
        shard, lse, coef_logp, coef_lse, target_local, vocab_start, real_vocab, has_lse_grad
    ):
        from rl_engine.kernels.ops.base import _C

        return _C.hip_deterministic_logp_backward(
            shard, lse, coef_logp, coef_lse, target_local, vocab_start, real_vocab, has_lse_grad
        )


class RocmVocabParallelLogprobOp(VocabParallelLogprobOp):
    """Contract-preserving ROCm implementation with HIP local reductions."""

    op_class = "logprob"
    is_batch_invariant = True
    use_native_tile_stats = True

    def apply(
        self,
        local_logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        contract: LogprobContract,
        tp_group: Any = None,
        num_vocab_tiles: int = DEFAULT_NUM_VOCAB_TILES,
        validate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _native_backward_available():
            raise RuntimeError(
                f"{BACKEND_ID} requires rl_engine._C built with a ROCm toolchain "
                "(hip_deterministic_logp_* symbols are missing); it does not fall back"
            )
        return apply_with_kernels(
            local_logits,
            target_ids,
            contract=contract,
            tp_group=tp_group,
            num_vocab_tiles=num_vocab_tiles,
            validate=validate,
            kernels=_HipKernels,
        )


__all__ = ["BACKEND_ID", "RocmVocabParallelLogprobOp"]
