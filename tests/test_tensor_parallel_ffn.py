# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""PR3 tensor-parallel Qwen-style FFN topology and invariance tests."""

from __future__ import annotations

import os
import queue
import socket
import tempfile
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSwiGLUOp
from rl_engine.kernels.ops.pytorch.ffn.tensor_parallel import (
    DeterministicTensorParallelCommunication,
    FFNContext,
    TensorParallelFFN,
    shard_qwen3_ffn_weights,
)


def _gloo_available() -> bool:
    return dist.is_available() and dist.is_gloo_available()


requires_gloo = pytest.mark.skipif(
    not _gloo_available(), reason="tensor-parallel FFN CPU test requires torch.distributed Gloo."
)


class _RecordingTPCommunication:
    """Gloo stand-in that makes the configured TP path observable."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int, int]] = []

    def all_reduce(self, tensor: torch.Tensor, *, ctx: FFNContext) -> torch.Tensor:
        assert ctx.tp_size is not None
        assert ctx.tp_rank is not None
        self.calls.append((tuple(tensor.shape), ctx.tp_size, ctx.tp_rank))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return tensor


def _configure_gloo_loopback() -> None:
    """Avoid hostname-resolution dependence in local CPU topology tests."""

    if "GLOO_SOCKET_IFNAME" in os.environ:
        return
    interfaces = {name for _, name in socket.if_nameindex()}
    loopback = "lo" if "lo" in interfaces else "lo0" if "lo0" in interfaces else None
    if loopback is not None:
        os.environ["GLOO_SOCKET_IFNAME"] = loopback


def _full_ffn(
    input_: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    swiglu = NativeSwiGLUOp()
    return F.linear(swiglu(F.linear(input_, gate_weight), F.linear(input_, up_weight)), down_weight)


def _fixed_row_gemm(input_2d: torch.Tensor, weight_2d: torch.Tensor) -> torch.Tensor:
    """CPU test adapter for the fixed-row contract of the det_gemm backends.

    Production uses the existing CUDA/Triton deterministic GEMM primitive via
    ``TensorParallelFFN(gemm=...)``.  CPU Gloo has no such backend, so this
    adapter invokes PyTorch once per output row; changing M therefore cannot
    select a different accumulation path for an already-valid row.
    """

    return torch.stack([torch.matmul(row.unsqueeze(0), weight_2d).squeeze(0) for row in input_2d])


def _tp_ffn_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_queue: Any,
    use_recording_tp_communication: bool = False,
) -> None:
    try:
        torch.set_num_threads(1)
        _configure_gloo_loopback()
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        tp_communication = (
            _RecordingTPCommunication() if use_recording_tp_communication else None
        )
        ctx = FFNContext(tp_group=dist.group.WORLD, tp_communication=tp_communication)
        assert ctx.tp_size == world_size
        assert ctx.tp_rank == rank

        hidden_size, intermediate_size = 8, 24
        generator = torch.Generator().manual_seed(20260812)
        input_base = torch.randn(3, 5, hidden_size, generator=generator)
        gate_full = torch.randn(intermediate_size, hidden_size, generator=generator)
        up_full = torch.randn(intermediate_size, hidden_size, generator=generator)
        down_full = torch.randn(hidden_size, intermediate_size, generator=generator)
        grad_output = torch.randn(3, 5, hidden_size, generator=generator)

        # Materialize the full reference before installing the collective spy.
        reference_input = input_base.detach().clone().requires_grad_(True)
        reference_gate = gate_full.detach().clone().requires_grad_(True)
        reference_up = up_full.detach().clone().requires_grad_(True)
        reference_down = down_full.detach().clone().requires_grad_(True)
        reference_output = _full_ffn(reference_input, reference_gate, reference_up, reference_down)
        reference_output.backward(grad_output)

        ffn = TensorParallelFFN.from_full_weights(
            gate_full, up_full, down_full, ctx=ctx, gemm=_fixed_row_gemm
        )
        tp_input = input_base.detach().clone().requires_grad_(True)

        # Forward uses one TP SUM for Down.  During backward the only TP SUM
        # is the combined Gate+Up dX at the replicated-input boundary.
        observed_collectives: list[tuple[int, ...]] = []
        original_all_reduce = dist.all_reduce

        def traced_all_reduce(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            observed_collectives.append(tuple(tensor.shape))
            return original_all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = traced_all_reduce  # type: ignore[assignment]
        try:
            output = ffn(tp_input)
            output.backward(grad_output)
        finally:
            dist.all_reduce = original_all_reduce  # type: ignore[assignment]

        local_intermediate = intermediate_size // world_size
        start = rank * local_intermediate
        stop = start + local_intermediate
        expected_collective_shape = tuple(input_base.shape)
        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "output_error": float((output - reference_output).abs().max().item()),
                "input_grad_error": float(
                    (tp_input.grad - reference_input.grad).abs().max().item()
                ),
                "gate_grad_error": float(
                    (ffn.gate_weight.grad - reference_gate.grad[start:stop]).abs().max().item()
                ),
                "up_grad_error": float(
                    (ffn.up_weight.grad - reference_up.grad[start:stop]).abs().max().item()
                ),
                "down_grad_error": float(
                    (ffn.down_weight.grad - reference_down.grad[:, start:stop]).abs().max().item()
                ),
                "collectives": observed_collectives,
                "expected_collective_shape": expected_collective_shape,
                "configured_collectives": (
                    [] if tp_communication is None else tp_communication.calls
                ),
            }
        )
    except Exception:
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _tp_batch_invariance_worker(
    rank: int, world_size: int, init_method: str, result_queue: Any
) -> None:
    try:
        torch.set_num_threads(1)
        _configure_gloo_loopback()
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        ctx = FFNContext(tp_group=dist.group.WORLD)
        hidden_size, intermediate_size = 8, 24
        generator = torch.Generator().manual_seed(99)
        input_valid = torch.randn(4, 3, hidden_size, generator=generator)
        padding = torch.randn(3, 3, hidden_size, generator=generator)
        gate_full = torch.randn(intermediate_size, hidden_size, generator=generator)
        up_full = torch.randn(intermediate_size, hidden_size, generator=generator)
        down_full = torch.randn(hidden_size, intermediate_size, generator=generator)
        ffn = TensorParallelFFN.from_full_weights(
            gate_full, up_full, down_full, ctx=ctx, gemm=_fixed_row_gemm
        )

        full_output = ffn(input_valid)
        single_output = ffn(input_valid[:1])
        padded_output = ffn(torch.cat((input_valid, padding), dim=0))
        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "slice_equal": bool(torch.equal(full_output[:1], single_output)),
                "padding_equal": bool(
                    torch.equal(full_output, padded_output[: input_valid.shape[0]])
                ),
            }
        )
    except Exception:
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_tp_workers(worker: Any, *worker_args: Any) -> list[dict[str, Any]]:
    world_size = 2
    mp_context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp_dir:
        init_file = Path(tmp_dir) / "tp_ffn_init"
        init_method = init_file.as_uri()
        result_queue = mp_context.Queue()
        workers = [
            mp_context.Process(
                target=worker,
                args=(rank, world_size, init_method, result_queue, *worker_args),
            )
            for rank in range(world_size)
        ]
        for process in workers:
            process.start()

        results: list[dict[str, Any]] = []
        try:
            for _ in workers:
                results.append(result_queue.get(timeout=30))
        except queue.Empty:
            pytest.fail("timed out waiting for tensor-parallel Gloo FFN workers")
        finally:
            for process in workers:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join()

        failures = [result for result in results if not result["ok"]]
        if failures:
            pytest.fail("\n".join(result["traceback"] for result in failures))
        failed_workers = [process for process in workers if process.exitcode != 0]
        if failed_workers:
            pytest.fail(
                f"tensor-parallel Gloo workers failed: {[p.exitcode for p in failed_workers]}"
            )
        return results


def _make_tp_cp_groups(rank: int) -> tuple[Any, Any]:
    """Create the fixed PR4 topology in identical order on every rank."""

    tp_group = None
    for ranks in ((0, 1), (2, 3)):
        group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            tp_group = group

    cp_group = None
    for ranks in ((0, 2), (1, 3)):
        group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            cp_group = group

    assert tp_group is not None
    assert cp_group is not None
    return tp_group, cp_group


def _tp_cp_ffn_worker(rank: int, world_size: int, init_method: str, result_queue: Any) -> None:
    try:
        torch.set_num_threads(1)
        _configure_gloo_loopback()
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        tp_group, cp_group = _make_tp_cp_groups(rank)
        ctx = FFNContext(tp_group=tp_group, cp_group=cp_group)
        assert (ctx.tp_size, ctx.cp_size) == (2, 2)
        assert ctx.tp_rank == rank % 2
        assert ctx.cp_rank == rank // 2

        hidden_size, intermediate_size = 8, 24
        batch_size, sequence_size = 3, 4
        generator = torch.Generator().manual_seed(20260813)
        input_full = torch.randn(batch_size, sequence_size, hidden_size, generator=generator)
        gate_full = torch.randn(intermediate_size, hidden_size, generator=generator)
        up_full = torch.randn(intermediate_size, hidden_size, generator=generator)
        down_full = torch.randn(hidden_size, intermediate_size, generator=generator)
        grad_output_full = torch.randn(batch_size, sequence_size, hidden_size, generator=generator)

        reference_input = input_full.detach().clone().requires_grad_(True)
        reference_gate = gate_full.detach().clone().requires_grad_(True)
        reference_up = up_full.detach().clone().requires_grad_(True)
        reference_down = down_full.detach().clone().requires_grad_(True)
        reference_output = _full_ffn(reference_input, reference_gate, reference_up, reference_down)
        reference_output.backward(grad_output_full)

        cp_index = rank // 2
        local_sequence = sequence_size // 2
        sequence_start = cp_index * local_sequence
        sequence_stop = sequence_start + local_sequence
        local_input = input_full[:, sequence_start:sequence_stop].detach().clone()
        local_input.requires_grad_(True)
        local_grad_output = grad_output_full[:, sequence_start:sequence_stop]
        ffn = TensorParallelFFN.from_full_weights(
            gate_full, up_full, down_full, ctx=ctx, gemm=_fixed_row_gemm
        )

        observed_collectives: list[tuple[str, tuple[int, ...]]] = []
        original_all_reduce = dist.all_reduce

        def traced_all_reduce(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            group = kwargs.get("group")
            if group is tp_group:
                group_name = "tp"
            elif group is cp_group:
                group_name = "cp"
            else:
                raise AssertionError("FFN used an unexpected process group for all_reduce.")
            observed_collectives.append((group_name, tuple(tensor.shape)))
            return original_all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = traced_all_reduce  # type: ignore[assignment]
        try:
            local_output = ffn(local_input)
            local_output.backward(local_grad_output)
        finally:
            dist.all_reduce = original_all_reduce  # type: ignore[assignment]

        gathered_outputs = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered_outputs, local_output)
        reconstructed_output = torch.cat((gathered_outputs[0], gathered_outputs[2]), dim=1)

        local_intermediate = intermediate_size // 2
        intermediate_start = (rank % 2) * local_intermediate
        intermediate_stop = intermediate_start + local_intermediate
        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "output_error": float(
                    (local_output - reference_output[:, sequence_start:sequence_stop])
                    .abs()
                    .max()
                    .item()
                ),
                "reconstructed_output_error": float(
                    (reconstructed_output - reference_output).abs().max().item()
                ),
                "input_grad_error": float(
                    (local_input.grad - reference_input.grad[:, sequence_start:sequence_stop])
                    .abs()
                    .max()
                    .item()
                ),
                "gate_grad_error": float(
                    (
                        ffn.gate_weight.grad
                        - reference_gate.grad[intermediate_start:intermediate_stop]
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "up_grad_error": float(
                    (ffn.up_weight.grad - reference_up.grad[intermediate_start:intermediate_stop])
                    .abs()
                    .max()
                    .item()
                ),
                "down_grad_error": float(
                    (
                        ffn.down_weight.grad
                        - reference_down.grad[:, intermediate_start:intermediate_stop]
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "collectives": observed_collectives,
                "activation_shape": tuple(local_input.shape),
                "gate_weight_shape": tuple(ffn.gate_weight.shape),
                "down_weight_shape": tuple(ffn.down_weight.shape),
            }
        )
    except Exception:
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _tp_cp_batch_invariance_worker(
    rank: int, world_size: int, init_method: str, result_queue: Any
) -> None:
    try:
        torch.set_num_threads(1)
        _configure_gloo_loopback()
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        tp_group, cp_group = _make_tp_cp_groups(rank)
        ctx = FFNContext(tp_group=tp_group, cp_group=cp_group)
        hidden_size, intermediate_size = 8, 24
        batch_size, sequence_size = 4, 4
        generator = torch.Generator().manual_seed(20260814)
        input_valid = torch.randn(batch_size, sequence_size, hidden_size, generator=generator)
        padding = torch.randn(3, sequence_size, hidden_size, generator=generator)
        gate_full = torch.randn(intermediate_size, hidden_size, generator=generator)
        up_full = torch.randn(intermediate_size, hidden_size, generator=generator)
        down_full = torch.randn(hidden_size, intermediate_size, generator=generator)
        ffn = TensorParallelFFN.from_full_weights(
            gate_full, up_full, down_full, ctx=ctx, gemm=_fixed_row_gemm
        )

        local_sequence = sequence_size // 2
        sequence_start = (rank // 2) * local_sequence
        sequence_stop = sequence_start + local_sequence
        valid_local = input_valid[:, sequence_start:sequence_stop]
        padded_local = torch.cat((input_valid, padding), dim=0)[:, sequence_start:sequence_stop]
        full_input = valid_local.detach().clone().requires_grad_(True)
        single_input = valid_local[:1].detach().clone().requires_grad_(True)
        padded_input = padded_local.detach().clone().requires_grad_(True)

        full_output = ffn(full_input)
        full_grad_output = torch.randn(full_output.shape, generator=generator)
        full_output.backward(full_grad_output)

        single_output = ffn(single_input)
        single_output.backward(full_grad_output[:1])

        padded_output = ffn(padded_input)
        padded_grad_output = torch.cat(
            (
                full_grad_output,
                torch.randn(
                    padded_output.shape[0] - batch_size,
                    *padded_output.shape[1:],
                    generator=generator,
                ),
            ),
            dim=0,
        )
        padded_output.backward(padded_grad_output)
        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "slice_equal": bool(torch.equal(full_output[:1], single_output)),
                "padding_equal": bool(torch.equal(full_output, padded_output[:batch_size])),
                "backward_slice_equal": bool(torch.equal(full_input.grad[:1], single_input.grad)),
                "backward_padding_equal": bool(
                    torch.equal(full_input.grad, padded_input.grad[:batch_size])
                ),
            }
        )
    except Exception:
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_tp_cp_workers(worker: Any) -> list[dict[str, Any]]:
    world_size = 4
    mp_context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp_dir:
        init_file = Path(tmp_dir) / "tp_cp_ffn_init"
        init_method = init_file.as_uri()
        result_queue = mp_context.Queue()
        workers = [
            mp_context.Process(target=worker, args=(rank, world_size, init_method, result_queue))
            for rank in range(world_size)
        ]
        for process in workers:
            process.start()

        results: list[dict[str, Any]] = []
        try:
            for _ in workers:
                results.append(result_queue.get(timeout=30))
        except queue.Empty:
            pytest.fail("timed out waiting for TP+CP Gloo FFN workers")
        finally:
            for process in workers:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join()

        failures = [result for result in results if not result["ok"]]
        if failures:
            pytest.fail("\n".join(result["traceback"] for result in failures))
        failed_workers = [process for process in workers if process.exitcode != 0]
        if failed_workers:
            pytest.fail(
                f"TP+CP Gloo workers failed: {[process.exitcode for process in failed_workers]}"
            )
        return results


@requires_gloo
def test_tensor_parallel_ffn_matches_full_reference_and_tp_backward_contract() -> None:
    """TP=2 FFN agrees with full FFN and reduces only at the documented sites."""

    results = _run_tp_workers(_tp_ffn_worker)
    for result in results:
        # The global reference contracts the full intermediate in one GEMM;
        # TP first rounds two local contractions and then sums them.  The
        # topology is therefore checked against the shared fp32 tolerance,
        # rather than requiring a different reduction tree to be bitwise equal.
        assert result["output_error"] <= 1e-4
        assert result["input_grad_error"] <= 1e-4
        assert result["gate_grad_error"] <= 1e-4
        assert result["up_grad_error"] <= 1e-4
        assert result["down_grad_error"] <= 1e-4
        # One forward Down reduction and one backward combined Gate+Up dX
        # reduction.  In particular, there is no [B, S, I / TP] all-reduce
        # for Down's dHidden.
        assert result["collectives"] == [
            result["expected_collective_shape"],
            result["expected_collective_shape"],
        ]


@requires_gloo
def test_tensor_parallel_ffn_routes_collectives_through_configured_communication() -> None:
    """Configured TP communication owns exactly the forward and backward SUMs."""

    results = _run_tp_workers(_tp_ffn_worker, True)
    for result in results:
        assert result["output_error"] <= 1e-4
        assert result["input_grad_error"] <= 1e-4
        assert result["collectives"] == [
            result["expected_collective_shape"],
            result["expected_collective_shape"],
        ]
        assert result["configured_collectives"] == [
            (result["expected_collective_shape"], 2, result["rank"]),
            (result["expected_collective_shape"], 2, result["rank"]),
        ]


@requires_gloo
def test_tensor_parallel_ffn_is_batch_invariant() -> None:
    """Valid rows are bitwise unchanged by TP FFN slicing or batch padding."""

    results = _run_tp_workers(_tp_batch_invariance_worker)
    assert all(result["slice_equal"] and result["padding_equal"] for result in results)


@requires_gloo
def test_tp_cp_ffn_matches_full_reference_and_cp_weight_gradient_contract() -> None:
    """TP=2, CP=2 matches the global FFN and reduces all three replica dWs."""

    results = _run_tp_cp_workers(_tp_cp_ffn_worker)
    for result in results:
        assert result["output_error"] <= 1e-4
        assert result["reconstructed_output_error"] <= 1e-4
        assert result["input_grad_error"] <= 1e-4
        assert result["gate_grad_error"] <= 1e-4
        assert result["up_grad_error"] <= 1e-4
        assert result["down_grad_error"] <= 1e-4

        tp_collectives = [shape for group, shape in result["collectives"] if group == "tp"]
        cp_collectives = [shape for group, shape in result["collectives"] if group == "cp"]
        # TP: Down forward and the combined Gate+Up dX backward. CP: one
        # replica-gradient reduction for each logical Down/Gate/Up parameter.
        assert tp_collectives == [result["activation_shape"], result["activation_shape"]]
        assert sorted(cp_collectives) == sorted(
            [
                result["down_weight_shape"],
                result["gate_weight_shape"],
                result["gate_weight_shape"],
            ]
        )
        assert len(result["collectives"]) == 5
        assert result["activation_shape"] not in cp_collectives


@requires_gloo
def test_tp_cp_ffn_is_batch_invariant() -> None:
    """TP+CP local rows and their input gradients are batch/padding invariant."""

    results = _run_tp_cp_workers(_tp_cp_batch_invariance_worker)
    assert all(
        result["slice_equal"]
        and result["padding_equal"]
        and result["backward_slice_equal"]
        and result["backward_padding_equal"]
        for result in results
    )


def test_ffn_context_rejects_unbound_multi_rank_tp() -> None:
    with pytest.raises(ValueError, match="explicit initialized tp_group"):
        FFNContext(tp_size=2)


def test_ffn_context_rejects_unbound_multi_rank_cp() -> None:
    with pytest.raises(ValueError, match="explicit initialized cp_group"):
        FFNContext(cp_size=2)


def test_qwen_weight_shards_follow_column_and_row_parallel_dimensions() -> None:
    ctx = FFNContext()
    gate = torch.empty(24, 8)
    up = torch.empty(24, 8)
    down = torch.empty(8, 24)
    gate_shard, up_shard, down_shard = shard_qwen3_ffn_weights(gate, up, down, ctx=ctx)
    assert gate_shard.shape == (24, 8)
    assert up_shard.shape == (24, 8)
    assert down_shard.shape == (8, 24)


def test_deterministic_tp_communication_rejects_cpu_tensors() -> None:
    communication = DeterministicTensorParallelCommunication()
    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        communication.all_reduce(torch.zeros(2), ctx=FFNContext())


def test_ffn_refuses_non_deterministic_default_gemm() -> None:
    ffn = TensorParallelFFN(8, 24)
    with pytest.raises(RuntimeError, match="explicit batch-invariant local GEMM"):
        ffn(torch.randn(2, 8))
