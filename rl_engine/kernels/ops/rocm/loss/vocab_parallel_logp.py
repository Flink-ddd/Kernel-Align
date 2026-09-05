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

``apply`` keeps the fused ROCm forward/backward local while reusing the shared
contract validation, rank-ordered transport, and fixed tile merge helpers.
``apply_with_entropy`` keeps the inherited reference autograd path because the
entropy gradient needs the full probability tensor anyway.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.kernels.logprob_contract import LogprobContract, LogprobContractError
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import (
    DEFAULT_NUM_VOCAB_TILES,
    VocabParallelLogprobOp,
    _gather_target_logit,
    _gather_tile_stats,
    _merge_tile_partials,
    _preflight_cross_rank_agreement,
    _tile_size,
    _validate_active_targets,
    _validate_invocation,
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


class _RocmVocabParallelLogprobFunction(torch.autograd.Function):
    """ROCm tile statistics and backward with the shared WS2 merge contract."""

    @staticmethod
    def forward(ctx, local_logits, target_1d, active_mask, contract, tp_group, tile):
        sharding = contract.sharding
        shard = local_logits.contiguous()
        local_tiles = sharding.local_vocab_size // tile
        if local_tiles <= 0:
            raise RuntimeError("native tile stats require at least one local vocab tile")
        local_m, local_s = _HipKernels.tile_stats(
            shard,
            sharding.local_vocab_start,
            sharding.real_vocab_size,
            local_tiles,
        )
        tile_counts = [
            (end - start) // tile for start, end in sharding.vocab_shard_bounds
        ]
        m_all, s_all = _gather_tile_stats(
            local_m.contiguous(),
            local_s.contiguous(),
            contract,
            tp_group,
            tile_counts,
        )
        safe_target = torch.where(active_mask, target_1d, torch.zeros_like(target_1d))
        target_logit = _gather_target_logit(
            shard, safe_target, contract, tp_group
        ).float()
        lse = _merge_tile_partials(m_all, s_all)
        selected_logp = torch.where(
            active_mask, target_logit - lse, torch.zeros_like(lse)
        )

        ctx.save_for_backward(shard, lse, safe_target, active_mask)
        ctx.local_vocab_start = sharding.local_vocab_start
        ctx.real_vocab_size = sharding.real_vocab_size
        ctx.set_materialize_grads(False)
        return selected_logp, lse

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        if not ctx.needs_input_grad[0] or (grad_logp is None and grad_lse is None):
            return None, None, None, None, None, None
        shard, lse, safe_target, active_mask = ctx.saved_tensors
        rows, local_vocab = shard.shape
        start = ctx.local_vocab_start
        if grad_logp is not None:
            coef_logp = (
                torch.where(active_mask, grad_logp, torch.zeros_like(grad_logp))
                .float()
                .contiguous()
            )
            owns = (safe_target >= start) & (safe_target < start + local_vocab)
            target_local = torch.where(
                owns & active_mask,
                safe_target - start,
                torch.full_like(safe_target, -1),
            ).contiguous()
        else:
            coef_logp = lse.new_zeros((rows,))
            target_local = torch.full(
                (rows,), -1, dtype=torch.long, device=shard.device
            )
        has_lse_grad = grad_lse is not None
        coef_lse = (
            grad_lse.float().contiguous()
            if has_lse_grad
            else lse.new_zeros((rows,))
        )
        grad = _HipKernels.backward(
            shard,
            lse.contiguous(),
            coef_logp,
            coef_lse,
            target_local,
            start,
            ctx.real_vocab_size,
            has_lse_grad,
        )
        return grad, None, None, None, None, None


def _apply_with_kernels(
    local_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    contract: LogprobContract,
    tp_group: Any,
    num_vocab_tiles: int,
    validate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(contract, LogprobContract):
        raise LogprobContractError("contract must be a LogprobContract")
    tile = _tile_size(contract, num_vocab_tiles)
    _validate_invocation(local_logits, target_ids, contract, tp_group)

    target_1d = target_ids.reshape(-1).to(
        device=local_logits.device, dtype=torch.long
    )
    active_mask = torch.tensor(
        contract.mask.active_mask,
        dtype=torch.bool,
        device=local_logits.device,
    )
    if validate:
        _validate_active_targets(
            target_1d, active_mask, contract.sharding.real_vocab_size
        )
        if contract.sharding.tp_world_size > 1:
            _preflight_cross_rank_agreement(
                contract, tp_group, num_vocab_tiles, True
            )

    selected_logp, lse = _RocmVocabParallelLogprobFunction.apply(
        local_logits, target_1d, active_mask, contract, tp_group, tile
    )
    if validate and bool((~torch.isfinite(lse) & active_mask).any().item()):
        raise LogprobContractError(
            "non-finite logsumexp on an active row: logits over the real "
            "vocabulary must be finite for every active token"
        )
    return selected_logp, lse


class RocmVocabParallelLogprobOp(VocabParallelLogprobOp):
    """Contract-preserving ROCm implementation with HIP local reductions."""

    op_class = "logprob"
    is_batch_invariant = True
    backend_id = BACKEND_ID

    def apply(
        self,
        local_logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        contract: LogprobContract,
        tp_group: Any = None,
        num_vocab_tiles: int = DEFAULT_NUM_VOCAB_TILES,
        validate: bool = True,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not deterministic:
            return super().apply(
                local_logits,
                target_ids,
                contract=contract,
                tp_group=tp_group,
                num_vocab_tiles=num_vocab_tiles,
                validate=validate,
                deterministic=False,
            )
        if not _native_backward_available():
            raise RuntimeError(
                f"{BACKEND_ID} requires rl_engine._C built with a ROCm toolchain "
                "(hip_deterministic_logp_* symbols are missing); it does not fall back"
            )
        return _apply_with_kernels(
            local_logits,
            target_ids,
            contract=contract,
            tp_group=tp_group,
            num_vocab_tiles=num_vocab_tiles,
            validate=validate,
        )


__all__ = ["BACKEND_ID", "RocmVocabParallelLogprobOp"]
