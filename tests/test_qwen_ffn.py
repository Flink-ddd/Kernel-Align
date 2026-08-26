# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the deterministic Qwen3 dense FFN.

Covers single-GPU correctness, token boundaries, Qwen3-8B shapes, TP/CP/SP
bitwise alignment, and DeterministicCollective cache lifetime.
"""

from __future__ import annotations

import queue
import tempfile
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import rl_engine.kernels.ops.pytorch.ffn.ffn as ffn_module
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.ffn.ffn import (
    QWEN3_8B_HIDDEN_SIZE,
    QWEN3_8B_INTERMEDIATE_SIZE,
    qwen3_ffn,
)

_REQUIRED_SYMBOLS = (
    "det_gemm_fwd",
    "det_gemm_fwd_rhs_transposed",
    "det_gemm_db_transposed",
    "swiglu_forward",
    "swiglu_backward",
)
_HIDDEN = 64
_INTERMEDIATE = 512
_TOPOLOGY_TOKENS = 256
_CP_TOKEN_COUNTS = (8, 32, 64, 96, 128, 256)
_TOKEN_BOUNDARY_COUNTS = (8, 31, 32, 33, 64, 96, 128)
_WORLD2_CONFIGS = (
    ("tp2_sp", 2, 1, True, _TOPOLOGY_TOKENS),
    ("cp2", 1, 2, False, _TOPOLOGY_TOKENS),
    *((f"cp2_T{token_count}", 1, 2, False, token_count) for token_count in _CP_TOKEN_COUNTS),
)
_WORLD4_CONFIGS = (
    ("tp4", 4, 1, False, _TOPOLOGY_TOKENS),
    ("cp4", 1, 4, False, _TOPOLOGY_TOKENS),
    ("tp2_cp2", 2, 2, False, _TOPOLOGY_TOKENS),
    ("tp2_cp2_sp", 2, 2, True, _TOPOLOGY_TOKENS),
    *((f"cp4_T{token_count}", 1, 4, False, token_count) for token_count in _CP_TOKEN_COUNTS),
)
_WORLD8_WORLD_GROUP_CONFIGS = (
    ("tp8", 8, 1, False, _TOPOLOGY_TOKENS),
    ("tp8_sp", 8, 1, True, _TOPOLOGY_TOKENS),
    ("cp8", 1, 8, False, _TOPOLOGY_TOKENS),
    *((f"cp8_T{token_count}", 1, 8, False, token_count) for token_count in _CP_TOKEN_COUNTS),
)
_WORLD8_TP2_CP4_CONFIGS = (("tp2_cp4", 2, 4, False, _TOPOLOGY_TOKENS),)
_WORLD8_TP4_CP2_CONFIGS = (
    ("tp4_cp2", 4, 2, False, _TOPOLOGY_TOKENS),
    ("tp4_cp2_sp", 4, 2, True, _TOPOLOGY_TOKENS),
)


def _has_sm90_ffn_devices(count: int) -> bool:
    return (
        _EXT_AVAILABLE
        and torch.distributed.is_available()
        and torch.distributed.is_nccl_available()
        and torch.cuda.device_count() >= count
        and all(torch.cuda.get_device_capability(index)[0] == 9 for index in range(count))
        and all(hasattr(_C, name) for name in _REQUIRED_SYMBOLS)
    )


def _has_sm90_ffn() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
        and _EXT_AVAILABLE
        and all(hasattr(_C, name) for name in _REQUIRED_SYMBOLS)
    )


requires_cuda_ffn = pytest.mark.skipif(
    not _has_sm90_ffn(),
    reason="FFN optimized-path validation requires SM90 and the GEMM/SwiGLU extension",
)


class _TorchKernelStub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def det_gemm_fwd(self, a, b):
        self.calls.append("det_gemm_fwd")
        return a @ b

    def det_gemm_fwd_rhs_transposed(self, a, bt):
        self.calls.append("det_gemm_fwd_rhs_transposed")
        return a @ bt.t()

    def det_gemm_db_transposed(self, a, grad_output):
        self.calls.append("det_gemm_db_transposed")
        return grad_output.t() @ a

    def swiglu_forward(self, gate, up):
        self.calls.append("swiglu_forward")
        return gate * torch.sigmoid(gate) * up

    def swiglu_backward(self, grad_output, gate, up):
        self.calls.append("swiglu_backward")
        sigmoid = torch.sigmoid(gate)
        grad_gate = grad_output * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_up = grad_output * gate * sigmoid
        return grad_gate, grad_up


def _reference(hidden_states, gate_weight, up_weight, down_weight):
    gate = hidden_states @ gate_weight.t()
    up = hidden_states @ up_weight.t()
    activated = F.silu(gate) * up
    return (activated @ down_weight.t()), gate, up, activated


def _randn(shape, *, seed, device="cpu", dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02
    return value.to(device=device, dtype=dtype)


def _close_ffn_collectives() -> None:
    for collective in list(ffn_module._COLLECTIVES.values()):
        collective.close()
    ffn_module._COLLECTIVES.clear()


def _shard_ranges(
    rank: int,
    *,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool,
    token_count: int,
    intermediate_size: int,
) -> tuple[int, int, int, int]:
    tp_rank = rank % tp_size
    cp_rank = rank // tp_size
    cp_tokens = token_count // cp_size
    local_tokens = cp_tokens // tp_size if sequence_parallel else cp_tokens
    token_start = cp_rank * cp_tokens
    if sequence_parallel:
        token_start += tp_rank * local_tokens
    token_end = token_start + local_tokens
    local_i = intermediate_size // tp_size
    feat_start = tp_rank * local_i
    feat_end = feat_start + local_i
    return token_start, token_end, feat_start, feat_end


def _spawn_nccl_workers(worker, world_size: int, worker_args=(), *, timeout: int = 180) -> None:
    if not _has_sm90_ffn_devices(world_size):
        pytest.skip(f"requires {world_size} SM90 GPUs, NCCL, and the GEMM/SwiGLU extension")

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "nccl_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=worker,
                args=(rank, world_size, init_method, result_queue, *worker_args),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        results = []
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=timeout))
        except queue.Empty:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail(f"timed out waiting for {world_size} FFN workers")
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()

    for result in sorted(results, key=lambda item: item["rank"]):
        assert result["ok"], result.get("traceback") or result.get("failures")
    for process in processes:
        assert process.exitcode == 0


def _distributed_ffn_backward_nccl_worker(
    rank,
    world_size,
    init_method,
    result_queue,
    cp_size,
    sequence_parallel,
):
    try:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        tp_size = world_size // cp_size
        tp_groups = [
            dist.new_group(list(range(cp_rank * tp_size, (cp_rank + 1) * tp_size)))
            for cp_rank in range(cp_size)
        ]
        cp_groups = [
            dist.new_group([cp_rank * tp_size + tp_rank for cp_rank in range(cp_size)])
            for tp_rank in range(tp_size)
        ]
        tp_rank = rank % tp_size
        cp_rank = rank // tp_size
        tp_group = tp_groups[cp_rank]
        cp_group = cp_groups[tp_rank] if cp_size > 1 else None

        token_count, hidden_size, intermediate_size = 8, 64, 128
        token_start, token_end, feature_start, feature_end = _shard_ranges(
            rank,
            tp_size=tp_size,
            cp_size=cp_size,
            sequence_parallel=sequence_parallel,
            token_count=token_count,
            intermediate_size=intermediate_size,
        )
        local_tokens = token_end - token_start

        device = torch.device("cuda", rank)
        rmsnorm_output = _randn(
            (token_count, hidden_size), seed=40, device=device, dtype=torch.bfloat16
        )
        gate_weight = _randn(
            (intermediate_size, hidden_size),
            seed=41,
            device=device,
            dtype=torch.bfloat16,
        )
        up_weight = _randn(
            (intermediate_size, hidden_size),
            seed=42,
            device=device,
            dtype=torch.bfloat16,
        )
        down_weight = _randn(
            (hidden_size, intermediate_size),
            seed=43,
            device=device,
            dtype=torch.bfloat16,
        )
        grad_output = _randn(
            (token_count, hidden_size), seed=44, device=device, dtype=torch.bfloat16
        )

        reference_inputs = [
            value.detach().float().requires_grad_(True)
            for value in (rmsnorm_output, gate_weight, up_weight, down_weight)
        ]
        reference_output, _, _, _ = _reference(*reference_inputs)
        reference_output.backward(grad_output.float())

        local_grad_output = grad_output[token_start:token_end].contiguous()
        actual_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (
                rmsnorm_output[token_start:token_end].contiguous(),
                gate_weight[feature_start:feature_end].contiguous(),
                up_weight[feature_start:feature_end].contiguous(),
                down_weight[:, feature_start:feature_end].contiguous(),
            )
        ]
        actual_output = qwen3_ffn(
            *actual_inputs,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
        )
        actual_output.backward(local_grad_output)

        expected_grads = (
            reference_inputs[0].grad[token_start:token_end],
            reference_inputs[1].grad[feature_start:feature_end],
            reference_inputs[2].grad[feature_start:feature_end],
            reference_inputs[3].grad[:, feature_start:feature_end],
        )
        torch.testing.assert_close(
            actual_output.float(),
            reference_output[token_start:token_end].detach(),
            atol=5e-2,
            rtol=2e-2,
        )
        for actual, expected in zip(actual_inputs, expected_grads, strict=True):
            torch.testing.assert_close(
                actual.grad.float(),
                expected,
                atol=5e-2,
                rtol=2e-2,
            )

        slice_size = max(1, local_tokens // 2)
        slice_start = (local_tokens - slice_size) // 2
        slice_end = slice_start + slice_size
        slice_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (
                actual_inputs[0][slice_start:slice_end].contiguous(),
                actual_inputs[1],
                actual_inputs[2],
                actual_inputs[3],
            )
        ]
        slice_output = qwen3_ffn(
            *slice_inputs,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
        )
        slice_output.backward(local_grad_output[slice_start:slice_end])

        assert torch.equal(
            slice_output,
            actual_output[slice_start:slice_end],
        ), "FFN output changed with the local token batch size"
        assert torch.equal(
            slice_inputs[0].grad,
            actual_inputs[0].grad[slice_start:slice_end],
        ), "FFN input gradient changed with the local token batch size"
        result_queue.put({"ok": True, "rank": rank})
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        _close_ffn_collectives()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _tp1_vs_tpn_train_infer_worker(rank, world_size, init_method, result_queue, expect_match):
    try:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        token_count, hidden_size, intermediate_size = 16, 64, 256
        hidden = _randn((token_count, hidden_size), seed=50, device=device, dtype=torch.bfloat16)
        gate_weight = _randn(
            (intermediate_size, hidden_size),
            seed=51,
            device=device,
            dtype=torch.bfloat16,
        )
        up_weight = _randn(
            (intermediate_size, hidden_size),
            seed=52,
            device=device,
            dtype=torch.bfloat16,
        )
        down_weight = _randn(
            (hidden_size, intermediate_size),
            seed=53,
            device=device,
            dtype=torch.bfloat16,
        )
        grad_output = _randn(
            (token_count, hidden_size), seed=54, device=device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            infer_tp1 = qwen3_ffn(hidden, gate_weight, up_weight, down_weight)

        tp1_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (hidden, gate_weight, up_weight, down_weight)
        ]
        train_tp1 = qwen3_ffn(*tp1_inputs)
        train_tp1.backward(grad_output)
        assert torch.equal(infer_tp1, train_tp1.detach()), "TP=1 train/infer forward mismatch"

        local_i = intermediate_size // world_size
        feat_start = rank * local_i
        feat_end = feat_start + local_i
        shard = (
            hidden,
            gate_weight[feat_start:feat_end].contiguous(),
            up_weight[feat_start:feat_end].contiguous(),
            down_weight[:, feat_start:feat_end].contiguous(),
        )
        with torch.no_grad():
            infer_tpn = qwen3_ffn(*shard, tp_group=dist.group.WORLD)

        tpn_inputs = [value.detach().clone().requires_grad_(True) for value in shard]
        train_tpn = qwen3_ffn(*tpn_inputs, tp_group=dist.group.WORLD)
        train_tpn.backward(grad_output)
        assert torch.equal(
            infer_tpn, train_tpn.detach()
        ), f"TP={world_size} train/infer forward mismatch"

        infer_match = torch.equal(infer_tp1, infer_tpn)
        train_match = torch.equal(train_tp1.detach(), train_tpn.detach())
        hidden_match = torch.equal(tp1_inputs[0].grad, tpn_inputs[0].grad)
        if expect_match:
            assert infer_match, f"TP=1 vs TP={world_size} infer forward mismatch"
            assert train_match, f"TP=1 vs TP={world_size} train forward mismatch"
            assert hidden_match, f"TP=1 vs TP={world_size} hidden grad mismatch"
        else:
            assert not infer_match, f"TP=1 vs TP={world_size} infer forward unexpectedly matched"
            assert not train_match, f"TP=1 vs TP={world_size} train forward unexpectedly matched"
            assert not hidden_match, f"TP=1 vs TP={world_size} hidden grad unexpectedly matched"

        assert torch.equal(
            tp1_inputs[1].grad[feat_start:feat_end], tpn_inputs[1].grad
        ), f"TP=1 vs TP={world_size} gate weight grad mismatch"
        assert torch.equal(
            tp1_inputs[2].grad[feat_start:feat_end], tpn_inputs[2].grad
        ), f"TP=1 vs TP={world_size} up weight grad mismatch"
        assert torch.equal(
            tp1_inputs[3].grad[:, feat_start:feat_end], tpn_inputs[3].grad
        ), f"TP=1 vs TP={world_size} down weight grad mismatch"

        result_queue.put({"ok": True, "rank": rank})
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        _close_ffn_collectives()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _make_topology_inputs(token_count, device):
    hidden = _randn((token_count, _HIDDEN), seed=60, device=device, dtype=torch.bfloat16)
    gate = _randn((_INTERMEDIATE, _HIDDEN), seed=61, device=device, dtype=torch.bfloat16)
    up = _randn((_INTERMEDIATE, _HIDDEN), seed=62, device=device, dtype=torch.bfloat16)
    down = _randn((_HIDDEN, _INTERMEDIATE), seed=63, device=device, dtype=torch.bfloat16)
    grad = _randn((token_count, _HIDDEN), seed=64, device=device, dtype=torch.bfloat16)
    return hidden, gate, up, down, grad


def _canonical(hidden, gate, up, down, grad):
    with torch.no_grad():
        infer = qwen3_ffn(hidden, gate, up, down)
    inputs = [value.detach().clone().requires_grad_(True) for value in (hidden, gate, up, down)]
    train = qwen3_ffn(*inputs)
    train.backward(grad)
    return infer, train, inputs


def _mesh_groups(dist, tp_size, cp_size):
    world_size = dist.get_world_size()
    if tp_size == world_size and cp_size == 1:
        return [dist.group.WORLD], []
    if cp_size == world_size and tp_size == 1:
        return [], [dist.group.WORLD]
    tp_groups = []
    if tp_size > 1:
        for cp_rank in range(cp_size):
            ranks = list(range(cp_rank * tp_size, (cp_rank + 1) * tp_size))
            tp_groups.append(dist.new_group(ranks))
    cp_groups = []
    if cp_size > 1:
        for tp_rank in range(tp_size):
            ranks = [cp_rank * tp_size + tp_rank for cp_rank in range(cp_size)]
            cp_groups.append(dist.new_group(ranks))
    return tp_groups, cp_groups


def _run_topology_config(
    rank,
    dist,
    meshes,
    *,
    name,
    tp_size,
    cp_size,
    sequence_parallel,
    hidden,
    gate,
    up,
    down,
    grad,
    infer_ref,
    train_ref,
    ref_inputs,
):
    key = (tp_size, cp_size)
    if key not in meshes:
        meshes[key] = _mesh_groups(dist, tp_size, cp_size)
    tp_groups, cp_groups = meshes[key]
    tp_rank = rank % tp_size
    cp_rank = rank // tp_size
    tp_group = tp_groups[cp_rank] if tp_size > 1 else None
    cp_group = cp_groups[tp_rank] if cp_size > 1 else None
    token_start, token_end, feat_start, feat_end = _shard_ranges(
        rank,
        tp_size=tp_size,
        cp_size=cp_size,
        sequence_parallel=sequence_parallel,
        token_count=hidden.size(0),
        intermediate_size=_INTERMEDIATE,
    )
    shard = (
        hidden[token_start:token_end].contiguous(),
        gate[feat_start:feat_end].contiguous(),
        up[feat_start:feat_end].contiguous(),
        down[:, feat_start:feat_end].contiguous(),
    )
    with torch.no_grad():
        infer = qwen3_ffn(
            *shard,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
        )
    inputs = [value.detach().clone().requires_grad_(True) for value in shard]
    train = qwen3_ffn(
        *inputs,
        tp_group=tp_group,
        cp_group=cp_group,
        sequence_parallel=sequence_parallel,
    )
    train.backward(grad[token_start:token_end].contiguous())

    assert torch.equal(infer, train.detach()), f"{name}: train/infer forward mismatch"
    assert torch.equal(
        infer, infer_ref[token_start:token_end]
    ), f"{name}: infer forward mismatch vs TP=1/CP=1"
    assert torch.equal(
        train.detach(), train_ref.detach()[token_start:token_end]
    ), f"{name}: train forward mismatch vs TP=1/CP=1"
    assert torch.equal(
        inputs[0].grad, ref_inputs[0].grad[token_start:token_end]
    ), f"{name}: hidden grad mismatch vs TP=1/CP=1"

    weight_checks = (
        (1, ref_inputs[1].grad[feat_start:feat_end], "gate"),
        (2, ref_inputs[2].grad[feat_start:feat_end], "up"),
        (3, ref_inputs[3].grad[:, feat_start:feat_end], "down"),
    )
    for index, expected, label in weight_checks:
        assert torch.equal(
            inputs[index].grad, expected
        ), f"{name}: {label} weight grad mismatch vs TP=1/CP=1"


def _topology_worker(rank, world_size, init_method, result_queue, configs):
    try:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5),
            device_id=torch.device("cuda", rank),
        )
        device = torch.device("cuda", rank)
        meshes = {}
        canonical = {}
        for name, tp_size, cp_size, sequence_parallel, token_count in configs:
            if token_count not in canonical:
                tensors = _make_topology_inputs(token_count, device)
                canonical[token_count] = (*tensors, *_canonical(*tensors))
            hidden, gate, up, down, grad, infer_ref, train_ref, ref_inputs = canonical[token_count]
            _run_topology_config(
                rank,
                dist,
                meshes,
                name=name,
                tp_size=tp_size,
                cp_size=cp_size,
                sequence_parallel=sequence_parallel,
                hidden=hidden,
                gate=gate,
                up=up,
                down=down,
                grad=grad,
                infer_ref=infer_ref,
                train_ref=train_ref,
                ref_inputs=ref_inputs,
            )
        result_queue.put({"ok": True, "rank": rank})
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        _close_ffn_collectives()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _ffn_tensors(token_count, device, seed, *, hidden=_HIDDEN, intermediate=_INTERMEDIATE):
    rmsnorm = _randn((token_count, hidden), seed=seed, device=device, dtype=torch.bfloat16)
    gate = _randn((intermediate, hidden), seed=seed + 1, device=device, dtype=torch.bfloat16)
    up = _randn((intermediate, hidden), seed=seed + 2, device=device, dtype=torch.bfloat16)
    down = _randn((hidden, intermediate), seed=seed + 3, device=device, dtype=torch.bfloat16)
    return rmsnorm, gate, up, down


def _cache_worker(rank, world_size, init_method, result_queue):
    try:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        ffn_module._COLLECTIVE_MIN_CAPACITY_BYTES = 64
        _close_ffn_collectives()

        small = _ffn_tensors(8, device, seed=100, intermediate=128)
        first = qwen3_ffn(*small, tp_group=dist.group.WORLD)
        assert len(ffn_module._COLLECTIVES) == 1
        ((cache_key, first_collective),) = ffn_module._COLLECTIVES.items()
        first_handle = first_collective._handle
        first_capacity = first_collective.max_size_bytes
        assert first_handle != 0

        repeated = qwen3_ffn(*small, tp_group=dist.group.WORLD)
        assert torch.equal(first, repeated)
        assert len(ffn_module._COLLECTIVES) == 1
        assert ffn_module._COLLECTIVES[cache_key] is first_collective
        assert first_collective._handle == first_handle

        large = _ffn_tensors(256, device, seed=110, intermediate=128)
        grown = qwen3_ffn(*large, tp_group=dist.group.WORLD)
        assert grown.shape[0] == 256
        assert len(ffn_module._COLLECTIVES) == 1
        grown_collective = next(iter(ffn_module._COLLECTIVES.values()))
        assert grown_collective is not first_collective
        assert first_collective._handle == 0
        assert grown_collective.max_size_bytes > first_capacity
        assert grown_collective._handle != 0

        _close_ffn_collectives()
        assert ffn_module._COLLECTIVES == {}
        recreated = qwen3_ffn(*small, tp_group=dist.group.WORLD)
        assert torch.equal(recreated, first)
        assert len(ffn_module._COLLECTIVES) == 1
        recreated_collective = next(iter(ffn_module._COLLECTIVES.values()))
        assert recreated_collective is not grown_collective
        assert recreated_collective._handle != 0

        rebuilt_group = dist.new_group(ranks=[0, 1])
        rebuilt = qwen3_ffn(*small, tp_group=rebuilt_group)
        assert torch.equal(rebuilt, first)
        assert len(ffn_module._COLLECTIVES) == 2

        _close_ffn_collectives()
        result_queue.put({"ok": True, "rank": rank})
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        _close_ffn_collectives()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _uneven_sp_worker(rank, world_size, init_method, result_queue):
    try:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        hidden, gate, up, down = _ffn_tensors(2, device, seed=130, intermediate=128)
        local_i = 128 // world_size
        feat_start = rank * local_i
        feat_end = feat_start + local_i
        local_hidden = hidden[:2] if rank == 0 else hidden[:1]
        try:
            qwen3_ffn(
                local_hidden,
                gate[feat_start:feat_end].contiguous(),
                up[feat_start:feat_end].contiguous(),
                down[:, feat_start:feat_end].contiguous(),
                tp_group=dist.group.WORLD,
                sequence_parallel=True,
            )
            result_queue.put(
                {
                    "ok": False,
                    "rank": rank,
                    "failures": "uneven SP tokens should have raised",
                }
            )
        except ValueError as exc:
            message = str(exc)
            if "matching shapes" not in message and "world_size" not in message:
                raise
            result_queue.put({"ok": True, "rank": rank})
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        _close_ffn_collectives()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _qwen3_8b_weights(device):
    hidden = _randn((8, QWEN3_8B_HIDDEN_SIZE), seed=90, device=device, dtype=torch.bfloat16)
    gate = _randn(
        (QWEN3_8B_INTERMEDIATE_SIZE, QWEN3_8B_HIDDEN_SIZE),
        seed=91,
        device=device,
        dtype=torch.bfloat16,
    )
    up = _randn(
        (QWEN3_8B_INTERMEDIATE_SIZE, QWEN3_8B_HIDDEN_SIZE),
        seed=92,
        device=device,
        dtype=torch.bfloat16,
    )
    down = _randn(
        (QWEN3_8B_HIDDEN_SIZE, QWEN3_8B_INTERMEDIATE_SIZE),
        seed=93,
        device=device,
        dtype=torch.bfloat16,
    )
    grad = _randn((8, QWEN3_8B_HIDDEN_SIZE), seed=94, device=device, dtype=torch.bfloat16)
    return hidden, gate, up, down, grad


def _qwen3_8b_tp2_worker(rank, world_size, init_method, result_queue):
    try:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        device = torch.device("cuda", rank)
        hidden, gate, up, down, grad = _qwen3_8b_weights(device)
        with torch.no_grad():
            infer_tp1 = qwen3_ffn(hidden, gate, up, down)
        tp1_inputs = [
            value.detach().clone().requires_grad_(True) for value in (hidden, gate, up, down)
        ]
        train_tp1 = qwen3_ffn(*tp1_inputs)
        train_tp1.backward(grad)

        local_i = QWEN3_8B_INTERMEDIATE_SIZE // world_size
        feat_start = rank * local_i
        feat_end = feat_start + local_i
        shard = (
            hidden,
            gate[feat_start:feat_end].contiguous(),
            up[feat_start:feat_end].contiguous(),
            down[:, feat_start:feat_end].contiguous(),
        )
        with torch.no_grad():
            infer_tp2 = qwen3_ffn(*shard, tp_group=dist.group.WORLD)
        tp2_inputs = [value.detach().clone().requires_grad_(True) for value in shard]
        train_tp2 = qwen3_ffn(*tp2_inputs, tp_group=dist.group.WORLD)
        train_tp2.backward(grad)

        assert torch.equal(infer_tp1, infer_tp2)
        assert torch.equal(train_tp1.detach(), train_tp2.detach())
        assert torch.equal(tp1_inputs[0].grad, tp2_inputs[0].grad)
        assert torch.equal(tp1_inputs[1].grad[feat_start:feat_end], tp2_inputs[1].grad)
        assert torch.equal(tp1_inputs[2].grad[feat_start:feat_end], tp2_inputs[2].grad)
        assert torch.equal(tp1_inputs[3].grad[:, feat_start:feat_end], tp2_inputs[3].grad)
        result_queue.put({"ok": True, "rank": rank})
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        _close_ffn_collectives()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_qwen_ffn_qwen3_8b_dimensions_are_pinned():
    assert QWEN3_8B_HIDDEN_SIZE == 4096
    assert QWEN3_8B_INTERMEDIATE_SIZE == 12288


def test_qwen_ffn_backward_matches_autograd_reference(monkeypatch):
    stub = _TorchKernelStub()
    monkeypatch.setattr(ffn_module, "_C", stub)
    monkeypatch.setattr(ffn_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(ffn_module, "_validate_ffn_inputs", lambda *args: None)

    hidden = _randn((2, 3, 8), seed=0)
    gate_weight = _randn((12, 8), seed=1)
    up_weight = _randn((12, 8), seed=2)
    down_weight = _randn((8, 12), seed=3)
    grad_output = _randn(hidden.shape, seed=4)

    ref_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected, _, _, _ = _reference(*ref_inputs)
    expected.backward(grad_output)

    actual_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    actual = qwen3_ffn(*actual_inputs)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected.detach())
    for actual_input, reference in zip(actual_inputs, ref_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, reference.grad)
    for weight in actual_inputs[1:]:
        assert weight.grad is not None
        assert weight.grad.is_contiguous()
        assert weight.grad.stride() == weight.stride()

    assert stub.calls.count("det_gemm_fwd") == 3
    assert stub.calls.count("det_gemm_fwd_rhs_transposed") == 3
    assert stub.calls.count("det_gemm_db_transposed") == 3
    assert stub.calls.count("swiglu_forward") == 1
    assert stub.calls.count("swiglu_backward") == 1


def test_qwen_ffn_disable_split_k_false_uses_torch_matmul(monkeypatch):
    stub = _TorchKernelStub()
    monkeypatch.setattr(ffn_module, "_C", stub)
    monkeypatch.setattr(ffn_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(ffn_module, "_validate_ffn_inputs", lambda *args: None)

    hidden = _randn((2, 3, 8), seed=0)
    gate_weight = _randn((12, 8), seed=1)
    up_weight = _randn((12, 8), seed=2)
    down_weight = _randn((8, 12), seed=3)
    grad_output = _randn(hidden.shape, seed=4)

    ref_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected, _, _, _ = _reference(*ref_inputs)
    expected.backward(grad_output)

    actual_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    actual = qwen3_ffn(*actual_inputs, disable_split_k=False)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected.detach())
    for actual_input, reference in zip(actual_inputs, ref_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, reference.grad)

    assert stub.calls.count("det_gemm_fwd") == 0
    assert stub.calls.count("det_gemm_fwd_rhs_transposed") == 0
    assert stub.calls.count("det_gemm_db_transposed") == 0
    assert stub.calls.count("swiglu_forward") == 1
    assert stub.calls.count("swiglu_backward") == 1


def test_qwen_ffn_deterministic_false_uses_production_gemm(monkeypatch):
    modes = []
    monkeypatch.setattr(ffn_module, "_validate_ffn_inputs", lambda *args: None)
    monkeypatch.setattr(ffn_module, "_require_ffn_kernels", lambda **kwargs: modes.append(kwargs))
    monkeypatch.setattr(
        ffn_module._DeterministicFFNFunction,
        "apply",
        lambda *args: args[-1],
    )

    tensors = [torch.empty(1)] * 4
    assert qwen3_ffn(*tensors, deterministic=False) is False
    assert modes == [{"disable_split_k": False}]


def test_qwen_ffn_rejects_conflicting_backend_switches():
    tensors = [torch.empty(1)] * 4

    with pytest.raises(ValueError, match="conflicting FFN backends"):
        qwen3_ffn(*tensors, deterministic=True, disable_split_k=False)


def test_qwen_ffn_rejects_non_bool_deterministic():
    tensors = [torch.empty(1)] * 4

    with pytest.raises(TypeError, match="deterministic must be a bool or None"):
        qwen3_ffn(*tensors, deterministic=1)  # type: ignore[arg-type]


def test_qwen_ffn_rejects_non_bool_disable_split_k():
    hidden = torch.empty((2, 8), dtype=torch.bfloat16)
    gate_weight = torch.empty((12, 8), dtype=torch.bfloat16)
    up_weight = torch.empty((12, 8), dtype=torch.bfloat16)
    down_weight = torch.empty((8, 12), dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="disable_split_k must be a bool"):
        qwen3_ffn(
            hidden,
            gate_weight,
            up_weight,
            down_weight,
            disable_split_k=1,  # type: ignore[arg-type]
        )


def test_qwen_ffn_rejects_non_huggingface_weight_layout():
    hidden = torch.empty((2, 8), dtype=torch.bfloat16)
    gate_weight = torch.empty((8, 12), dtype=torch.bfloat16)
    up_weight = torch.empty((12, 8), dtype=torch.bfloat16)
    down_weight = torch.empty((8, 12), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="gate_weight must have shape"):
        qwen3_ffn(hidden, gate_weight, up_weight, down_weight)


@requires_cuda_ffn
@pytest.mark.parametrize("disable_split_k", [True, False])
def test_qwen_ffn_cuda_forward_backward_matches_fp32_reference(disable_split_k):
    hidden = _randn((2, 3, 64), seed=10, device="cuda", dtype=torch.bfloat16)
    gate_weight = _randn((128, 64), seed=11, device="cuda", dtype=torch.bfloat16)
    up_weight = _randn((128, 64), seed=12, device="cuda", dtype=torch.bfloat16)
    down_weight = _randn((64, 128), seed=13, device="cuda", dtype=torch.bfloat16)
    grad_output = _randn(hidden.shape, seed=14, device="cuda", dtype=torch.bfloat16)

    ref_inputs = [
        value.detach().cpu().float().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected, _, _, _ = _reference(*ref_inputs)
    expected.backward(grad_output.cpu().float())

    actual_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    actual = qwen3_ffn(*actual_inputs, disable_split_k=disable_split_k)
    actual.backward(grad_output)

    torch.testing.assert_close(
        actual.cpu().float(),
        expected.detach(),
        atol=5e-2,
        rtol=2e-2,
    )
    for actual_input, reference in zip(actual_inputs, ref_inputs, strict=True):
        torch.testing.assert_close(
            actual_input.grad.cpu().float(),
            reference.grad,
            atol=5e-2,
            rtol=2e-2,
        )


@requires_cuda_ffn
def test_qwen_ffn_cuda_forward_and_input_gradient_are_batch_invariant():
    gate_weight = _randn((128, 64), seed=20, device="cuda", dtype=torch.bfloat16)
    up_weight = _randn((128, 64), seed=21, device="cuda", dtype=torch.bfloat16)
    down_weight = _randn((64, 128), seed=22, device="cuda", dtype=torch.bfloat16)
    hidden = _randn((6, 64), seed=23, device="cuda", dtype=torch.bfloat16)
    grad_output = _randn(hidden.shape, seed=24, device="cuda", dtype=torch.bfloat16)

    full_hidden = hidden.detach().clone().requires_grad_(True)
    full_output = qwen3_ffn(full_hidden, gate_weight, up_weight, down_weight)
    full_output.backward(grad_output)

    slice_hidden = hidden[2:4].detach().clone().requires_grad_(True)
    slice_output = qwen3_ffn(slice_hidden, gate_weight, up_weight, down_weight)
    slice_output.backward(grad_output[2:4])

    assert torch.equal(slice_output, full_output[2:4])
    assert torch.equal(slice_hidden.grad, full_hidden.grad[2:4])


@requires_cuda_ffn
def test_qwen_ffn_cuda_train_and_infer_forward_are_bitwise_identical():
    hidden = _randn((16, 64), seed=30, device="cuda", dtype=torch.bfloat16)
    gate_weight = _randn((128, 64), seed=31, device="cuda", dtype=torch.bfloat16)
    up_weight = _randn((128, 64), seed=32, device="cuda", dtype=torch.bfloat16)
    down_weight = _randn((64, 128), seed=33, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        infer = qwen3_ffn(hidden, gate_weight, up_weight, down_weight)
    train_hidden = hidden.detach().clone().requires_grad_(True)
    train = qwen3_ffn(train_hidden, gate_weight, up_weight, down_weight)
    assert torch.equal(infer, train.detach())


@requires_cuda_ffn
@pytest.mark.parametrize("token_count", _TOKEN_BOUNDARY_COUNTS)
def test_qwen_ffn_output_and_hidden_grad_are_batch_invariant(token_count):
    device = torch.device("cuda", 0)
    gate = _randn((256, _HIDDEN), seed=70, device=device, dtype=torch.bfloat16)
    up = _randn((256, _HIDDEN), seed=71, device=device, dtype=torch.bfloat16)
    down = _randn((_HIDDEN, 256), seed=72, device=device, dtype=torch.bfloat16)
    hidden = _randn((token_count, _HIDDEN), seed=73, device=device, dtype=torch.bfloat16)
    grad = _randn((token_count, _HIDDEN), seed=74, device=device, dtype=torch.bfloat16)

    full_hidden = hidden.detach().clone().requires_grad_(True)
    full_output = qwen3_ffn(full_hidden, gate, up, down)
    full_output.backward(grad)
    slice_end = min(8, token_count)
    slice_hidden = hidden[:slice_end].detach().clone().requires_grad_(True)
    slice_output = qwen3_ffn(slice_hidden, gate, up, down)
    slice_output.backward(grad[:slice_end])

    assert torch.equal(slice_output, full_output[:slice_end])
    assert torch.equal(slice_hidden.grad, full_hidden.grad[:slice_end])


@requires_cuda_ffn
def test_qwen_ffn_qwen3_8b_shapes_run_and_are_batch_invariant():
    device = torch.device("cuda", 0)
    hidden, gate, up, down, grad = _qwen3_8b_weights(device)

    with torch.no_grad():
        infer = qwen3_ffn(hidden, gate, up, down)
    full_hidden = hidden.detach().clone().requires_grad_(True)
    full_gate = gate.detach().clone().requires_grad_(True)
    full_up = up.detach().clone().requires_grad_(True)
    full_down = down.detach().clone().requires_grad_(True)
    train = qwen3_ffn(full_hidden, full_gate, full_up, full_down)
    train.backward(grad)
    assert torch.equal(infer, train.detach())

    slice_hidden = hidden[:4].detach().clone().requires_grad_(True)
    slice_out = qwen3_ffn(slice_hidden, full_gate, full_up, full_down)
    slice_out.backward(grad[:4])
    assert torch.equal(slice_out, train.detach()[:4])
    assert torch.equal(slice_hidden.grad, full_hidden.grad[:4])


@requires_cuda_ffn
def test_qwen_ffn_sequence_parallel_requires_tensor_parallel_group():
    device = torch.device("cuda", 0)
    hidden, gate, up, down = _ffn_tensors(8, device, seed=120, intermediate=128)
    with pytest.raises(ValueError, match="sequence_parallel requires a tensor-parallel group"):
        qwen3_ffn(hidden, gate, up, down, sequence_parallel=True)


def test_qwen_ffn_tp_correctness_and_batch_invariance():
    _spawn_nccl_workers(_distributed_ffn_backward_nccl_worker, 2, (1, False), timeout=90)


def test_qwen_ffn_tp_sp_correctness_and_batch_invariance():
    _spawn_nccl_workers(_distributed_ffn_backward_nccl_worker, 2, (1, True), timeout=90)


def test_qwen_ffn_tp_cp_correctness_and_batch_invariance():
    _spawn_nccl_workers(_distributed_ffn_backward_nccl_worker, 4, (2, False), timeout=90)


def test_qwen_ffn_tp_cp_sp_correctness_and_batch_invariance():
    _spawn_nccl_workers(_distributed_ffn_backward_nccl_worker, 4, (2, True), timeout=90)


def test_qwen_ffn_tp1_vs_tp2_train_infer_bitwise_identical():
    _spawn_nccl_workers(_tp1_vs_tpn_train_infer_worker, 2, (True,), timeout=120)


def test_qwen_ffn_tp1_vs_tp8_train_infer_bitwise_identical():
    _spawn_nccl_workers(_tp1_vs_tpn_train_infer_worker, 8, (True,), timeout=120)


def test_qwen_ffn_world2_tp_sp_and_cp_match_tp1_cp1_bitwise():
    _spawn_nccl_workers(_topology_worker, 2, (_WORLD2_CONFIGS,), timeout=120)


def test_qwen_ffn_world4_tp_cp_sp_match_tp1_cp1_bitwise():
    _spawn_nccl_workers(_topology_worker, 4, (_WORLD4_CONFIGS,), timeout=120)


def test_qwen_ffn_world8_tp8_and_cp8_match_tp1_cp1_bitwise():
    _spawn_nccl_workers(_topology_worker, 8, (_WORLD8_WORLD_GROUP_CONFIGS,), timeout=120)


def test_qwen_ffn_world8_tp2_cp4_match_tp1_cp1_bitwise():
    _spawn_nccl_workers(_topology_worker, 8, (_WORLD8_TP2_CP4_CONFIGS,), timeout=120)


def test_qwen_ffn_world8_tp4_cp2_match_tp1_cp1_bitwise():
    _spawn_nccl_workers(_topology_worker, 8, (_WORLD8_TP4_CP2_CONFIGS,), timeout=120)


def test_qwen_ffn_collective_cache_reuses_grows_closes_and_rebuilds_group():
    _spawn_nccl_workers(_cache_worker, 2, timeout=120)


def test_qwen_ffn_sequence_parallel_rejects_uneven_tokens():
    _spawn_nccl_workers(_uneven_sp_worker, 2, timeout=90)


def test_qwen_ffn_qwen3_8b_shapes_tp2_matches_tp1_bitwise():
    _spawn_nccl_workers(_qwen3_8b_tp2_worker, 2, timeout=180)
