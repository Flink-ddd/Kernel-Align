# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic vocab-parallel TP selected-token logprob reference (issue #241 PR3).

Implements the WS2 contract in ``rl_engine.kernels.logprob_contract`` with a
TP-independent vocab tile decomposition: the padded vocabulary is split into
``num_vocab_tiles`` fixed tiles, every tile's fp32 ``(max, sumexp)`` partial is
computed from a contiguous ``[n, tile]`` tensor, all tile partials travel by
all-gather (transport only), and every rank merges them in global tile-index
order over a fixed ``[n, num_vocab_tiles]`` shape.  The TP degree only decides
which rank computes which tiles and never changes any floating-point grouping,
so outputs and gradients are bitwise-identical across TP degrees
(``DeterminismScope.CROSS_TP_BITWISE``) as long as ``num_vocab_tiles`` is held
fixed.  A fixed per-shard merge order alone cannot provide this property:
shard boundaries would regroup the combines differently at each degree.

Consequences of the tile structure:

- ``num_vocab_tiles`` is part of the numerical identity.  It must be pinned
  across ranks (enforced by the preflight) and across the TP degrees being
  compared; it is never derived from the shard layout.
- Every shard boundary must be tile-aligned; misalignment fails loudly.
- At TP=1 the result matches the WS1 ``NativeBatchInvariantLogpOp`` only
  within the #108 logprob tolerance, not bitwise — the WS1 op reduces the
  whole ``[n, V]`` row at once, which groups the sums differently.

Preconditions: logits over the real vocabulary must be finite.  A row whose
real-vocab logits are all ``-inf`` has no finite logsumexp; with
``validate=True`` such a row fails loudly if it is active.

The selected logprob is zero-filled at inactive rows (``MaskSpec.active_mask``
is the sole authority; with validation enabled an active row can never legally
hold ``ignore_index``).  The vocab-domain LSE is returned for every row and is
differentiable everywhere, including inactive rows.

``deterministic=False`` trades the guarantee for speed: each rank reduces its
whole shard in one pass and the per-shard partials are merged in shard order,
so the floating-point grouping changes with the TP degree and nothing is
promised about reproducibility.  ``num_vocab_tiles`` is ignored (no tile
alignment is required), and a contract declaring
``determinism_scope=cross_tp_bitwise`` is rejected loudly — the fast path
cannot honor it.  Batch invariance is unaffected: rows never mix either way.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.kernels.logprob_contract import (
    DeterminismScope,
    LogprobContract,
    LogprobContractError,
    LogprobDType,
)

BACKEND_ID = "pytorch-vocab-parallel-logp-ws2"
DEFAULT_NUM_VOCAB_TILES = 64

_TORCH_TO_CONTRACT_DTYPE = {
    torch.bfloat16: LogprobDType.BF16,
    torch.float16: LogprobDType.FP16,
    torch.float32: LogprobDType.FP32,
}


def _require_distributed_initialized():
    import torch.distributed as dist

    if not dist.is_available():
        raise LogprobContractError("vocab-parallel logprob requires torch.distributed.")
    if not dist.is_initialized():
        raise LogprobContractError(
            "vocab-parallel logprob requires an initialized process group when "
            "the contract declares tp_world_size > 1."
        )
    return dist


def _tile_size(contract: LogprobContract, num_vocab_tiles: int) -> int:
    if isinstance(num_vocab_tiles, bool) or not isinstance(num_vocab_tiles, int):
        raise LogprobContractError(
            f"num_vocab_tiles must be a positive integer; got {num_vocab_tiles!r}"
        )
    if num_vocab_tiles <= 0:
        raise LogprobContractError(
            f"num_vocab_tiles must be a positive integer; got {num_vocab_tiles}"
        )
    padded = contract.sharding.padded_vocab_size
    if padded % num_vocab_tiles != 0:
        raise LogprobContractError(
            f"num_vocab_tiles={num_vocab_tiles} must divide " f"padded_vocab_size={padded} exactly"
        )
    tile = padded // num_vocab_tiles
    for rank, (start, end) in enumerate(contract.sharding.vocab_shard_bounds):
        if start % tile != 0 or end % tile != 0:
            raise LogprobContractError(
                f"vocab_shard_bounds[{rank}]=[{start}, {end}) is not aligned to the "
                f"vocab tile size {tile} (num_vocab_tiles={num_vocab_tiles}); "
                "cross-TP bitwise determinism requires tile-aligned shard bounds"
            )
    return tile


def _validate_invocation(
    local_logits: torch.Tensor,
    target_ids: torch.Tensor,
    contract: LogprobContract,
    tp_group: Any,
) -> None:
    if not isinstance(contract, LogprobContract):
        raise LogprobContractError("contract must be a LogprobContract")
    if local_logits.dim() != 2:
        raise LogprobContractError(
            f"local_logits must be 2-D [num_tokens, local_vocab]; got {local_logits.dim()}-D"
        )
    if target_ids.dim() != 1 or target_ids.shape[0] != local_logits.shape[0]:
        raise LogprobContractError(
            f"target_ids must be 1-D with one entry per token; got shape "
            f"{tuple(target_ids.shape)} for {local_logits.shape[0]} tokens"
        )
    sharding = contract.sharding
    if local_logits.shape[1] != sharding.local_vocab_size:
        raise LogprobContractError(
            f"local_logits has {local_logits.shape[1]} vocab columns but the contract "
            f"declares local shard [{sharding.local_vocab_start}, "
            f"{sharding.local_vocab_end}) of size {sharding.local_vocab_size}"
        )
    if local_logits.shape[0] != contract.mask.num_tokens:
        raise LogprobContractError(
            f"local_logits has {local_logits.shape[0]} tokens but MaskSpec declares "
            f"num_tokens={contract.mask.num_tokens}"
        )
    declared = _TORCH_TO_CONTRACT_DTYPE.get(local_logits.dtype)
    if declared is not contract.dtype:
        raise LogprobContractError(
            f"local_logits dtype {local_logits.dtype} does not match the contract "
            f"dtype {contract.dtype.value}"
        )
    if sharding.tp_world_size > 1:
        dist = _require_distributed_initialized()
        group_rank = dist.get_rank(group=tp_group)
        group_world = dist.get_world_size(group=tp_group)
        if group_world != sharding.tp_world_size:
            raise LogprobContractError(
                f"tp_group world size {group_world} does not match the contract "
                f"tp_world_size={sharding.tp_world_size}; pass the TP subgroup, "
                "not the global group"
            )
        if group_rank != sharding.tp_rank:
            raise LogprobContractError(
                f"tp_group rank {group_rank} does not match the contract "
                f"tp_rank={sharding.tp_rank}"
            )


def _validate_active_targets(
    target_1d: torch.Tensor, active_mask: torch.Tensor, real_vocab_size: int
) -> None:
    bad = active_mask & ((target_1d < 0) | (target_1d >= real_vocab_size))
    if bool(bad.any().item()):
        bad_values = target_1d[bad]
        raise LogprobContractError(
            "active target_ids must lie in the real vocabulary "
            f"[0, {real_vocab_size}); got values in "
            f"[{int(bad_values.min().item())}, {int(bad_values.max().item())}] "
            "on active rows"
        )


def _preflight_cross_rank_agreement(
    contract: LogprobContract, tp_group: Any, num_vocab_tiles: int, deterministic: bool
) -> None:
    """All-gather (fingerprint, backend id, tile count, mode) and abort on mismatch.

    The tile count travels as ``None`` when ``deterministic=False``: the fast
    path never uses it, so ranks must not fail preflight over an irrelevant
    value — but they must never disagree on the mode itself, or they would
    issue different collectives.
    """

    dist = _require_distributed_initialized()
    payload = (
        contract.cross_rank_fingerprint(),
        BACKEND_ID,
        int(num_vocab_tiles) if deterministic else None,
        bool(deterministic),
    )
    world = dist.get_world_size(group=tp_group)
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, payload, group=tp_group)
    mismatched = [(rank, other) for rank, other in enumerate(gathered) if other != payload]
    if mismatched:
        rank, other = mismatched[0]
        raise LogprobContractError(
            "cross-rank preflight failed: rank "
            f"{contract.sharding.tp_rank} has {payload} but rank {rank} has {other}; "
            "all TP ranks must agree on the contract fingerprint, backend id, "
            "num_vocab_tiles, and deterministic mode before any collective"
        )


def _local_tile_stats(z_masked: torch.Tensor, tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 per-tile ``(max, sumexp)`` partials for this rank's shard.

    Each tile is reduced as a contiguous ``[n, tile]`` tensor so the reduction
    shape and layout are identical no matter which rank computes the tile or
    what the local shard size is.  An all-``-inf`` (padding-only) tile yields
    the identity partial ``(-inf, 0)`` without evaluating ``exp(-inf - (-inf))``.
    """

    n, local_vocab = z_masked.shape
    m_parts: list[torch.Tensor] = []
    s_parts: list[torch.Tensor] = []
    for tile_index in range(local_vocab // tile):
        block = z_masked[:, tile_index * tile : (tile_index + 1) * tile].contiguous()
        m_t = block.max(dim=-1).values
        finite = m_t > float("-inf")
        m_safe = torch.where(finite, m_t, torch.zeros_like(m_t))
        s_t = (block - m_safe.unsqueeze(-1)).exp().sum(dim=-1)
        s_t = torch.where(finite, s_t, torch.zeros_like(s_t))
        m_parts.append(m_t)
        s_parts.append(s_t)
    return torch.stack(m_parts, dim=1), torch.stack(s_parts, dim=1)


def _gather_tile_stats(
    local_m: torch.Tensor,
    local_s: torch.Tensor,
    contract: LogprobContract,
    tp_group: Any,
    tile_counts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble every rank's partials in global shard order.

    ``tile_counts`` holds each rank's partial count: one per tile on the
    deterministic path, exactly one per shard on the fast path.
    """

    sharding = contract.sharding
    if sharding.tp_world_size == 1:
        return local_m.contiguous(), local_s.contiguous()

    dist = _require_distributed_initialized()
    n = local_m.shape[0]
    max_tiles = max(tile_counts)
    packed = local_m.new_zeros((n, max_tiles, 2))
    packed[:, : local_m.shape[1], 0] = local_m
    packed[:, : local_s.shape[1], 1] = local_s
    packed = packed.contiguous()
    gathered = [torch.empty_like(packed) for _ in range(sharding.tp_world_size)]
    dist.all_gather(gathered, packed, group=tp_group)

    m_parts = [gathered[rank][:, : tile_counts[rank], 0] for rank in range(len(tile_counts))]
    s_parts = [gathered[rank][:, : tile_counts[rank], 1] for rank in range(len(tile_counts))]
    return torch.cat(m_parts, dim=1).contiguous(), torch.cat(s_parts, dim=1).contiguous()


def _gather_target_logit(
    z_masked: torch.Tensor,
    safe_target: torch.Tensor,
    contract: LogprobContract,
    tp_group: Any,
) -> torch.Tensor:
    """Exact selected-target logit via a select-by-owner copy."""

    sharding = contract.sharding
    n = z_masked.shape[0]
    start = sharding.local_vocab_start
    local_vocab = sharding.local_vocab_size
    local_idx = (safe_target - start).clamp(0, max(local_vocab - 1, 0))
    owns = (safe_target >= start) & (safe_target < sharding.local_vocab_end)
    rows = torch.arange(n, device=z_masked.device)
    local_contrib = torch.where(
        owns, z_masked[rows, local_idx], torch.zeros_like(safe_target, dtype=z_masked.dtype)
    ).contiguous()

    if sharding.tp_world_size == 1:
        stacked = local_contrib.unsqueeze(0)
    else:
        dist = _require_distributed_initialized()
        gathered = [torch.empty_like(local_contrib) for _ in range(sharding.tp_world_size)]
        dist.all_gather(gathered, local_contrib, group=tp_group)
        stacked = torch.stack(gathered, dim=0)

    starts = torch.tensor(
        [bound_start for bound_start, _ in sharding.vocab_shard_bounds],
        device=safe_target.device,
        dtype=torch.long,
    )
    owner = torch.bucketize(safe_target, starts, right=True) - 1
    return stacked[owner, rows]


def _merge_tile_partials(m_all: torch.Tensor, s_all: torch.Tensor) -> torch.Tensor:
    """Fixed-order (max, sumexp) merge over [n, num_vocab_tiles]."""

    M = m_all.max(dim=1).values
    finite = M > float("-inf")
    M_safe = torch.where(finite, M, torch.zeros_like(M))
    terms = s_all * (m_all - M_safe.unsqueeze(1)).exp()
    S = terms.sum(dim=1)
    return M + S.log()


class _VocabParallelLogprobFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_logits, target_1d, active_mask, contract, tp_group, tile):
        z_masked = local_logits.float()
        sharding = contract.sharding
        global_ids = torch.arange(
            sharding.local_vocab_start, sharding.local_vocab_end, device=z_masked.device
        )
        padding_cols = global_ids >= sharding.real_vocab_size
        if bool(padding_cols.any()):
            z_masked = z_masked.masked_fill(padding_cols.unsqueeze(0), float("-inf"))

        safe_target = torch.where(active_mask, target_1d, torch.zeros_like(target_1d))

        if tile is not None:
            local_m, local_s = _local_tile_stats(z_masked, tile)
            tile_counts = [(end - start) // tile for start, end in sharding.vocab_shard_bounds]
        else:
            # Fast path: one (max, sumexp) partial over the whole shard. The
            # grouping now depends on the shard layout, so no cross-TP claim.
            local_m, local_s = _local_tile_stats(z_masked, z_masked.shape[1])
            tile_counts = [1] * sharding.tp_world_size
        m_all, s_all = _gather_tile_stats(local_m, local_s, contract, tp_group, tile_counts)
        target_logit = _gather_target_logit(z_masked, safe_target, contract, tp_group)
        lse = _merge_tile_partials(m_all, s_all)

        selected_logp = torch.where(active_mask, target_logit - lse, torch.zeros_like(lse))

        ctx.save_for_backward(z_masked, lse, safe_target, active_mask, padding_cols)
        ctx.local_vocab_start = sharding.local_vocab_start
        ctx.local_vocab_size = sharding.local_vocab_size
        ctx.input_dtype = local_logits.dtype
        ctx.set_materialize_grads(False)
        return selected_logp, lse

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        if not ctx.needs_input_grad[0] or (grad_logp is None and grad_lse is None):
            return None, None, None, None, None, None

        z_masked, lse, safe_target, active_mask, padding_cols = ctx.saved_tensors
        n, local_vocab = z_masked.shape
        finite_row = torch.isfinite(lse)
        lse_safe = torch.where(finite_row, lse, torch.zeros_like(lse))
        p = (z_masked - lse_safe.unsqueeze(1)).exp()
        p = torch.where(finite_row.unsqueeze(1), p, torch.zeros_like(p))

        grad = torch.zeros_like(z_masked)
        if grad_logp is not None:
            local_idx = (safe_target - ctx.local_vocab_start).clamp(0, max(local_vocab - 1, 0))
            owns = (safe_target >= ctx.local_vocab_start) & (
                safe_target < ctx.local_vocab_start + local_vocab
            )
            onehot = torch.zeros_like(z_masked)
            hit = owns & active_mask
            rows = torch.arange(n, device=z_masked.device)[hit]
            onehot[rows, local_idx[hit]] = 1.0
            g_logp = torch.where(active_mask, grad_logp, torch.zeros_like(grad_logp))
            grad = grad + g_logp.unsqueeze(1) * (onehot - p)
        if grad_lse is not None:
            grad = grad + grad_lse.unsqueeze(1) * p
        if bool(padding_cols.any()):
            grad = grad.masked_fill(padding_cols.unsqueeze(0), 0.0)
        return grad.to(ctx.input_dtype), None, None, None, None, None


class VocabParallelLogprobOp:
    """Vocab-parallel selected-token logprob (WS2 reference).

    Deterministic by default (cross-TP bitwise, tile-ordered merge);
    ``deterministic=False`` selects the faster whole-shard reduction with no
    reproducibility guarantee.
    """

    op_class = "logprob"
    is_batch_invariant = True

    def __init__(self) -> None:
        pass

    def __call__(
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
        return self.apply(
            local_logits,
            target_ids,
            contract=contract,
            tp_group=tp_group,
            num_vocab_tiles=num_vocab_tiles,
            validate=validate,
            deterministic=deterministic,
        )

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
        if not isinstance(contract, LogprobContract):
            raise LogprobContractError("contract must be a LogprobContract")
        if not isinstance(deterministic, bool):
            raise LogprobContractError(f"deterministic must be a bool; got {deterministic!r}")
        if deterministic:
            tile = _tile_size(contract, num_vocab_tiles)
        elif contract.reduction.determinism_scope is DeterminismScope.CROSS_TP_BITWISE:
            raise LogprobContractError(
                "deterministic=False cannot honor determinism_scope=cross_tp_bitwise: "
                "the fast path reduces each shard in one piece, so its floating-point "
                "grouping changes with the TP degree; keep deterministic=True or relax "
                "the contract to determinism_scope=fixed_topology"
            )
        else:
            tile = None
        _validate_invocation(local_logits, target_ids, contract, tp_group)

        target_1d = target_ids.reshape(-1).to(device=local_logits.device, dtype=torch.long)
        active_mask = torch.tensor(
            contract.mask.active_mask, dtype=torch.bool, device=local_logits.device
        )
        if validate:
            _validate_active_targets(target_1d, active_mask, contract.sharding.real_vocab_size)
            if contract.sharding.tp_world_size > 1:
                _preflight_cross_rank_agreement(contract, tp_group, num_vocab_tiles, deterministic)

        selected_logp, lse = _VocabParallelLogprobFunction.apply(
            local_logits, target_1d, active_mask, contract, tp_group, tile
        )

        if validate and bool((~torch.isfinite(lse) & active_mask).any().item()):
            raise LogprobContractError(
                "non-finite logsumexp on an active row: logits over the real "
                "vocabulary must be finite for every active token"
            )
        return selected_logp, lse


__all__ = [
    "BACKEND_ID",
    "DEFAULT_NUM_VOCAB_TILES",
    "VocabParallelLogprobOp",
]
