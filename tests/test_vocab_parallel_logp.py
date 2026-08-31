# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic vocab-parallel TP logprob reference tests (issue #241 PR3).

Bit-level determinism assertions compare raw bit patterns via
``tensor.view(torch.int32)`` rather than ``torch.equal``: value equality
treats ``-0.0 == 0.0`` as equal and ``NaN != NaN`` as different, neither of
which is what a bitwise claim means.
"""

from __future__ import annotations

import queue
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from rl_engine.kernels.gtest.tolerance import load_contract
from rl_engine.kernels.logprob_contract import (
    DeterminismScope,
    LogprobContract,
    LogprobContractError,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.ops.pytorch.loss.batch_invariant_logp import NativeBatchInvariantLogpOp
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import (
    BACKEND_ID,
    VocabParallelLogprobOp,
)
from rl_engine.kernels.registry import KernelRegistry, OpBackend

REAL_VOCAB = 27
PADDED_VOCAB = 32
NUM_TILES = 8
NUM_TOKENS = 6
ACTIVE = (True, True, True, True, True, False)


def _even_bounds(padded: int, world: int) -> tuple[tuple[int, int], ...]:
    shard = padded // world
    return tuple(
        (rank * shard, padded if rank == world - 1 else (rank + 1) * shard) for rank in range(world)
    )


def _contract(
    *,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    bounds: tuple[tuple[int, int], ...] | None = None,
    real_vocab: int = REAL_VOCAB,
    padded_vocab: int = PADDED_VOCAB,
    num_tokens: int = NUM_TOKENS,
    active: tuple[bool, ...] = ACTIVE,
    dtype: str = "fp32",
    determinism_scope: str = "cross_tp_bitwise",
) -> LogprobContract:
    return LogprobContract(
        role="train",
        dtype=dtype,
        mask=MaskSpec(num_tokens=num_tokens, active_mask=active),
        sharding=ShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            vocab_shard_bounds=(
                bounds if bounds is not None else _even_bounds(padded_vocab, tp_world_size)
            ),
            real_vocab_size=real_vocab,
            padded_vocab_size=padded_vocab,
        ),
        reduction=ReductionSpec(determinism_scope=determinism_scope),
    )


def _inputs(dtype=torch.float32, seed: int = 2026):
    torch.manual_seed(seed)
    logits = torch.randn(NUM_TOKENS, PADDED_VOCAB, dtype=torch.float32).to(dtype)
    targets = torch.tensor([1, 5, REAL_VOCAB - 1, 0, 13, -100])
    return logits, targets


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    view_dtype = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[tensor.dtype]
    return tensor.contiguous().view(view_dtype)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool((_bits(a) == _bits(b)).all())


def _case_shard_size_mismatch():
    logits, targets = _inputs()
    return logits, targets, _contract(tp_rank=0, tp_world_size=2), NUM_TILES, "vocab columns"


def _case_mask_length_mismatch():
    logits, targets = _inputs()
    contract = _contract(num_tokens=NUM_TOKENS + 1, active=ACTIVE + (True,))
    return logits, targets, contract, NUM_TILES, "num_tokens"


def _case_dtype_mismatch():
    logits, targets = _inputs()
    return logits, targets, _contract(dtype="bf16"), NUM_TILES, "dtype"


def _case_tile_misaligned_bounds():
    # Tile size is 32/8 = 4; a boundary at 6 is misaligned.
    logits, targets = _inputs()
    contract = _contract(tp_world_size=2, bounds=((0, 6), (6, 32)))
    return logits[:, :6], targets, contract, NUM_TILES, "tile"


def _case_bad_num_vocab_tiles():
    logits, targets = _inputs()
    return logits, targets, _contract(), 7, "num_vocab_tiles"


def _case_active_target_out_of_real_vocab():
    logits, targets = _inputs()
    bad_targets = targets.clone()
    bad_targets[0] = REAL_VOCAB  # padding column, active row
    return logits, bad_targets, _contract(), NUM_TILES, "real vocabulary"


def _case_all_inf_active_row():
    logits, targets = _inputs()
    poisoned = logits.clone()
    poisoned[0, :] = float("-inf")
    return poisoned, targets, _contract(), NUM_TILES, "non-finite"


@pytest.mark.parametrize(
    "case",
    [
        _case_shard_size_mismatch,
        _case_mask_length_mismatch,
        _case_dtype_mismatch,
        _case_tile_misaligned_bounds,
        _case_bad_num_vocab_tiles,
        _case_active_target_out_of_real_vocab,
        _case_all_inf_active_row,
    ],
    ids=lambda fn: fn.__name__.removeprefix("_case_"),
)
def test_invalid_invocations_fail_loudly(case):
    logits, targets, contract, num_tiles, match = case()
    with pytest.raises(LogprobContractError, match=match):
        VocabParallelLogprobOp()(logits, targets, contract=contract, num_vocab_tiles=num_tiles)


class TestSingleRank:
    def test_repeated_runs_are_bitwise_identical(self):
        contract = _contract()
        logits, targets = _inputs()
        op = VocabParallelLogprobOp()
        logp_a, lse_a = op(logits, targets, contract=contract, num_vocab_tiles=NUM_TILES)
        logp_b, lse_b = op(logits, targets, contract=contract, num_vocab_tiles=NUM_TILES)
        assert _bitwise_equal(logp_a, logp_b)
        assert _bitwise_equal(lse_a, lse_b)

    def test_batch_invariance_same_row_any_context(self):
        contract_full = _contract()
        logits, targets = _inputs()
        op = VocabParallelLogprobOp()
        logp_full, lse_full = op(logits, targets, contract=contract_full, num_vocab_tiles=NUM_TILES)

        contract_single = _contract(num_tokens=1, active=(True,))
        logp_one, lse_one = op(
            logits[2:3], targets[2:3], contract=contract_single, num_vocab_tiles=NUM_TILES
        )
        assert _bitwise_equal(logp_full[2:3], logp_one)
        assert _bitwise_equal(lse_full[2:3], lse_one)

    def test_matches_ws1_batch_invariant_logp_within_contract_tolerance(self):
        tolerance = load_contract()["accuracy"]["default"]["logprob"]["float32"]
        contract = _contract(padded_vocab=REAL_VOCAB + 5)
        # Use a real==padded contract so the WS1 op sees identical logits.
        contract = _contract(real_vocab=PADDED_VOCAB, padded_vocab=PADDED_VOCAB)
        logits, targets = _inputs()
        logp, _ = VocabParallelLogprobOp()(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        ws1 = NativeBatchInvariantLogpOp().apply(logits, targets)
        active = torch.tensor(ACTIVE)
        assert torch.allclose(
            logp[active], ws1[active], atol=tolerance["atol"], rtol=tolerance["rtol"]
        )

    def test_padding_columns_are_excluded_and_finite(self):
        contract = _contract()
        logits, targets = _inputs()
        boosted = logits.clone()
        boosted[:, REAL_VOCAB:] = 1e4  # huge padding logits must not leak into LSE
        logp, lse = VocabParallelLogprobOp()(
            boosted, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        ref_lse = torch.logsumexp(boosted[:, :REAL_VOCAB].float(), dim=-1)
        assert torch.isfinite(logp).all() and torch.isfinite(lse).all()
        assert torch.allclose(lse, ref_lse, atol=1e-5)

    def test_inactive_rows_zero_filled_lse_still_exported(self):
        contract = _contract()
        logits, targets = _inputs()
        logp, lse = VocabParallelLogprobOp()(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        assert logp[-1].item() == 0.0
        assert torch.isfinite(lse[-1])


class TestNonDeterministicPath:
    """deterministic=False: the fast whole-shard reduction with no guarantee."""

    def test_rejects_a_cross_tp_bitwise_contract(self):
        logits, targets = _inputs()
        with pytest.raises(LogprobContractError, match="cross_tp_bitwise"):
            VocabParallelLogprobOp()(logits, targets, contract=_contract(), deterministic=False)

    def test_rejects_a_non_bool_flag(self):
        logits, targets = _inputs()
        with pytest.raises(LogprobContractError, match="deterministic must be a bool"):
            VocabParallelLogprobOp()(logits, targets, contract=_contract(), deterministic=1)

    def test_matches_the_deterministic_path_within_tolerance(self):
        tolerance = load_contract()["accuracy"]["default"]["logprob"]["float32"]
        logits, targets = _inputs()
        relaxed = _contract(determinism_scope="fixed_topology")
        logp_fast, lse_fast = VocabParallelLogprobOp()(
            logits, targets, contract=relaxed, deterministic=False
        )
        logp_det, lse_det = VocabParallelLogprobOp()(
            logits, targets, contract=_contract(), num_vocab_tiles=NUM_TILES
        )
        assert torch.allclose(logp_fast, logp_det, atol=tolerance["atol"], rtol=tolerance["rtol"])
        assert torch.allclose(lse_fast, lse_det, atol=tolerance["atol"], rtol=tolerance["rtol"])

    def test_ignores_tile_constraints(self):
        # 7 does not divide the padded vocab; the deterministic path rejects it,
        # the fast path never looks at it.
        logits, targets = _inputs()
        relaxed = _contract(determinism_scope="fixed_topology")
        logp, lse = VocabParallelLogprobOp()(
            logits, targets, contract=relaxed, num_vocab_tiles=7, deterministic=False
        )
        ref_lse = torch.logsumexp(logits[:, :REAL_VOCAB].float(), dim=-1)
        assert torch.allclose(lse, ref_lse, atol=1e-5)
        assert torch.isfinite(logp).all()

    def test_grads_match_autograd_oracle(self):
        tolerance = load_contract()["accuracy"]["default"]["logprob"]["float32"]
        relaxed = _contract(determinism_scope="fixed_topology")
        logits, targets = _inputs()
        x = logits.clone().requires_grad_(True)
        logp, lse = VocabParallelLogprobOp()(x, targets, contract=relaxed, deterministic=False)
        (logp.sum() + 0.5 * lse.sum()).backward()

        y = logits.clone().requires_grad_(True)
        ref_lse = torch.logsumexp(y[:, :REAL_VOCAB].float(), dim=-1)
        safe = targets.clamp(0, REAL_VOCAB - 1)
        ref_logp = y[torch.arange(NUM_TOKENS), safe].float() - ref_lse
        ref_logp = torch.where(torch.tensor(ACTIVE), ref_logp, torch.zeros_like(ref_logp))
        (ref_logp.sum() + 0.5 * ref_lse.sum()).backward()

        assert torch.allclose(x.grad, y.grad, atol=tolerance["atol"], rtol=tolerance["rtol"])
        assert bool((x.grad[:, REAL_VOCAB:] == 0).all())


class TestBackward:
    def test_grads_match_autograd_oracle(self):
        tolerance = load_contract()["accuracy"]["default"]["logprob"]["float32"]
        contract = _contract()
        logits, targets = _inputs()
        x = logits.clone().requires_grad_(True)
        logp, lse = VocabParallelLogprobOp()(
            x, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        (logp.sum() + 0.5 * lse.sum()).backward()

        y = logits.clone().requires_grad_(True)
        ref_lse = torch.logsumexp(y[:, :REAL_VOCAB].float(), dim=-1)
        safe = targets.clamp(0, REAL_VOCAB - 1)
        ref_logp = y[torch.arange(NUM_TOKENS), safe].float() - ref_lse
        ref_logp = torch.where(torch.tensor(ACTIVE), ref_logp, torch.zeros_like(ref_logp))
        (ref_logp.sum() + 0.5 * ref_lse.sum()).backward()

        assert torch.allclose(x.grad, y.grad, atol=tolerance["atol"], rtol=tolerance["rtol"])
        assert bool((x.grad[:, REAL_VOCAB:] == 0).all())

        # No grad requested -> outputs detached from autograd entirely.
        logp_ng, lse_ng = VocabParallelLogprobOp()(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        assert not logp_ng.requires_grad and not lse_ng.requires_grad

    def test_inactive_rows_grad_asymmetry(self):
        """The logp term is zeroed on inactive rows; the lse term still flows —
        lse is a row property exported (and differentiable) for every row."""

        contract = _contract()
        logits, targets = _inputs()

        x = logits.clone().requires_grad_(True)
        _, lse = VocabParallelLogprobOp()(x, targets, contract=contract, num_vocab_tiles=NUM_TILES)
        lse.sum().backward()
        assert bool((x.grad[-1, :REAL_VOCAB].abs() > 0).any())

        z = logits.clone().requires_grad_(True)
        logp, _ = VocabParallelLogprobOp()(z, targets, contract=contract, num_vocab_tiles=NUM_TILES)
        logp.sum().backward()
        assert bool((z.grad[-1] == 0).all())

    def test_entropy_matches_full_vocab_oracle_and_backpropagates(self):
        contract = _contract()
        logits, targets = _inputs()
        x = logits.clone().requires_grad_(True)
        logp, _lse, entropy = VocabParallelLogprobOp().apply_with_entropy(
            x,
            targets,
            contract=contract,
            num_vocab_tiles=NUM_TILES,
        )

        y = logits.clone().requires_grad_(True)
        ref_log_probs = torch.log_softmax(y[:, :REAL_VOCAB].float(), dim=-1)
        safe = targets.clamp(0, REAL_VOCAB - 1)
        ref_logp = ref_log_probs[torch.arange(NUM_TOKENS), safe]
        ref_logp = torch.where(torch.tensor(ACTIVE), ref_logp, torch.zeros_like(ref_logp))
        ref_entropy = -(ref_log_probs.exp() * ref_log_probs).sum(dim=-1)

        torch.testing.assert_close(entropy, ref_entropy)
        (logp.sum() + entropy.sum()).backward()
        (ref_logp.sum() + ref_entropy.sum()).backward()
        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-5)


def test_dispatch_resolves_reference_and_leaves_legacy_untouched():
    registry = KernelRegistry()
    contract = _contract()

    result = registry.get_logprob_op(contract)
    assert result.capability.backend_id == BACKEND_ID
    assert result.provenance["fallback"] is False
    assert isinstance(result.op, VocabParallelLogprobOp)
    assert (
        result.provenance["contract"]["reduction"]["determinism_scope"]
        == DeterminismScope.CROSS_TP_BITWISE.value
    )

    by_id = registry.get_logprob_op(contract, requested_backend=BACKEND_ID)
    assert by_id.capability.backend_id == BACKEND_ID
    by_kind = registry.get_logprob_op(contract, requested_backend="reference")
    assert by_kind.capability.backend_id == BACKEND_ID

    for ops in registry._priority_map.values():
        for candidates in ops.values():
            assert OpBackend.PYTORCH_VOCAB_PARALLEL_LOGP not in candidates


# ---------------------------------------------------------------------------
# Multi-rank gloo tests (spawn pattern from tests/test_linear_logp.py)
# ---------------------------------------------------------------------------


def _gloo_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_gloo_available()


requires_gloo = pytest.mark.skipif(
    not _gloo_available(), reason="requires torch.distributed with the gloo backend"
)

_WORLD_SIZE = 4
_UNEVEN_BOUNDS = ((0, 4), (4, 16), (16, 24), (24, 32))  # tile-aligned (tile=4)


def _tp_worker(rank, world_size, init_method, result_queue, scenario):
    import torch.distributed as dist

    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            backend="gloo", init_method=init_method, rank=rank, world_size=world_size
        )
        dtype = torch.bfloat16 if scenario == "bf16" else torch.float32
        dtype_name = "bf16" if scenario == "bf16" else "fp32"
        bounds = _UNEVEN_BOUNDS if scenario == "uneven" else _even_bounds(PADDED_VOCAB, world_size)
        logits, targets = _inputs(dtype=dtype)
        start, end = bounds[rank]

        op = VocabParallelLogprobOp()
        tiles = 16 if scenario == "preflight" and rank == 0 else NUM_TILES
        contract_tp = _contract(
            tp_rank=rank, tp_world_size=world_size, bounds=bounds, dtype=dtype_name
        )

        if scenario == "preflight":
            try:
                op(
                    logits[:, start:end].contiguous().clone(),
                    targets,
                    contract=contract_tp,
                    tp_group=dist.group.WORLD,
                    num_vocab_tiles=tiles,
                )
                result_queue.put({"ok": False, "rank": rank, "traceback": "no error raised"})
            except LogprobContractError:
                result_queue.put({"ok": True, "rank": rank})
            return

        if scenario == "mode_mismatch":
            # Same relaxed contract everywhere; only rank 0 runs deterministic.
            relaxed = _contract(
                tp_rank=rank,
                tp_world_size=world_size,
                bounds=bounds,
                determinism_scope="fixed_topology",
            )
            try:
                op(
                    logits[:, start:end].contiguous().clone(),
                    targets,
                    contract=relaxed,
                    tp_group=dist.group.WORLD,
                    num_vocab_tiles=NUM_TILES,
                    deterministic=(rank == 0),
                )
                result_queue.put({"ok": False, "rank": rank, "traceback": "no error raised"})
            except LogprobContractError:
                result_queue.put({"ok": True, "rank": rank})
            return

        if scenario == "nondeterministic":
            relaxed = _contract(
                tp_rank=rank,
                tp_world_size=world_size,
                bounds=bounds,
                determinism_scope="fixed_topology",
            )
            shard = logits[:, start:end].contiguous().clone().requires_grad_(True)
            logp_tp, lse_tp = op(
                shard, targets, contract=relaxed, tp_group=dist.group.WORLD, deterministic=False
            )
            (logp_tp.sum() + 0.5 * lse_tp.sum()).backward()

            full = logits.clone().requires_grad_(True)
            relaxed_tp1 = _contract(determinism_scope="fixed_topology")
            logp_one, lse_one = op(full, targets, contract=relaxed_tp1, deterministic=False)
            (logp_one.sum() + 0.5 * lse_one.sum()).backward()

            result_queue.put(
                {
                    "ok": True,
                    "rank": rank,
                    "logp_close": torch.allclose(logp_tp, logp_one, atol=1e-6),
                    "lse_close": torch.allclose(lse_tp, lse_one, atol=1e-6),
                    "grad_close": torch.allclose(shard.grad, full.grad[:, start:end], atol=1e-6),
                    "logp": logp_tp.detach().float(),
                    "lse": lse_tp.detach().float(),
                }
            )
            return

        shard = logits[:, start:end].contiguous().clone().requires_grad_(True)
        logp_tp, lse_tp = op(
            shard,
            targets,
            contract=contract_tp,
            tp_group=dist.group.WORLD,
            num_vocab_tiles=NUM_TILES,
        )
        (logp_tp.sum() + 0.5 * lse_tp.sum()).backward()

        # In-process TP=1 run of the same op on the full logits: the cross-TP
        # bitwise claim is TP=n output == TP=1 output, bit for bit.
        full = logits.clone().requires_grad_(True)
        contract_tp1 = _contract(dtype=dtype_name)
        logp_one, lse_one = op(full, targets, contract=contract_tp1, num_vocab_tiles=NUM_TILES)
        (logp_one.sum() + 0.5 * lse_one.sum()).backward()

        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "logp_bits_match": _bitwise_equal(logp_tp, logp_one),
                "lse_bits_match": _bitwise_equal(lse_tp, lse_one),
                "grad_bits_match": _bitwise_equal(shard.grad, full.grad[:, start:end]),
                "logp": logp_tp.detach().float(),
                "lse": lse_tp.detach().float(),
            }
        )
    except Exception:
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_gloo_scenario(scenario):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "gloo_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_tp_worker,
                args=(rank, _WORLD_SIZE, init_method, result_queue, scenario),
            )
            for rank in range(_WORLD_SIZE)
        ]
        results = []
        try:
            for process in processes:
                process.start()
            for _ in range(_WORLD_SIZE):
                try:
                    results.append(result_queue.get(timeout=60))
                except queue.Empty:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    pytest.fail("timed out waiting for vocab-parallel gloo workers")
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()
    results.sort(key=lambda item: item["rank"])
    for result in results:
        assert result["ok"], result.get("traceback")
    for process in processes:
        assert process.exitcode == 0
    return results


@requires_gloo
@pytest.mark.parametrize("scenario", ["even", "uneven", "bf16"])
def test_tp4_bitwise_identical_to_tp1(scenario):
    results = _run_gloo_scenario(scenario)
    for result in results:
        assert result["logp_bits_match"], f"rank {result['rank']} logp bits differ from TP=1"
        assert result["lse_bits_match"], f"rank {result['rank']} lse bits differ from TP=1"
        assert result["grad_bits_match"], f"rank {result['rank']} grad bits differ from TP=1"
    # Outputs are replicated: every rank must hold identical bits.
    for other in results[1:]:
        assert _bitwise_equal(results[0]["logp"], other["logp"])
        assert _bitwise_equal(results[0]["lse"], other["lse"])


@requires_gloo
def test_preflight_rejects_mismatched_num_vocab_tiles():
    _run_gloo_scenario("preflight")


@requires_gloo
def test_preflight_rejects_mixed_deterministic_modes():
    _run_gloo_scenario("mode_mismatch")


@requires_gloo
def test_tp4_nondeterministic_matches_tp1_within_tolerance():
    """No bitwise claim: the fast path's grouping changes with the TP degree.
    The values must still agree within fp32 tolerance, and the outputs stay
    replicated bitwise across ranks — every rank merges the same partials."""

    results = _run_gloo_scenario("nondeterministic")
    for result in results:
        assert result["logp_close"], f"rank {result['rank']} logp differs from TP=1 beyond atol"
        assert result["lse_close"], f"rank {result['rank']} lse differs from TP=1 beyond atol"
        assert result["grad_close"], f"rank {result['rank']} grads differ from TP=1 beyond atol"
    for other in results[1:]:
        assert _bitwise_equal(results[0]["logp"], other["logp"])
        assert _bitwise_equal(results[0]["lse"], other["lse"])
