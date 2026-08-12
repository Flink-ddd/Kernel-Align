# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the deterministic Qwen3 dense FFN backward assembly."""

from __future__ import annotations

import queue
import tempfile
import traceback
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
    qwen3_ffn_backward,
)

_REQUIRED_SYMBOLS = (
    "det_gemm_fwd",
    "det_gemm_db",
    "swiglu_forward",
    "swiglu_backward",
)
_HAS_SM90_FFN = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 9
    and _EXT_AVAILABLE
    and all(hasattr(_C, name) for name in _REQUIRED_SYMBOLS)
)

requires_cuda_ffn = pytest.mark.skipif(
    not _HAS_SM90_FFN,
    reason="FFN optimized-path validation requires SM90 and the GEMM/SwiGLU extension symbols",
)


def _gloo_available():
    return torch.distributed.is_available() and torch.distributed.is_gloo_available()


requires_gloo = pytest.mark.skipif(
    not _gloo_available(),
    reason="tensor-parallel FFN backward CPU test requires torch.distributed Gloo",
)


class _TorchKernelStub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def det_gemm_fwd(self, a, b):
        self.calls.append("det_gemm_fwd")
        return a @ b

    def det_gemm_db(self, a, grad_output):
        self.calls.append("det_gemm_db")
        return a.t().contiguous() @ grad_output

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


def _tp_ffn_backward_gloo_worker(rank, world_size, init_method, result_queue):
    try:
        import torch.distributed as dist

        torch.set_num_threads(1)
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        stub = _TorchKernelStub()
        ffn_module._C = stub
        ffn_module._EXT_AVAILABLE = True
        ffn_module._validate_ffn_backward_inputs = lambda *args: None

        token_count, hidden_size, intermediate_size = 6, 5, 12
        local_intermediate = intermediate_size // world_size
        shard_start = rank * local_intermediate
        shard_end = shard_start + local_intermediate

        rmsnorm_output = _randn((token_count, hidden_size), seed=30)
        gate_weight = _randn((intermediate_size, hidden_size), seed=31)
        up_weight = _randn((intermediate_size, hidden_size), seed=32)
        down_weight = _randn((hidden_size, intermediate_size), seed=33)
        grad_output = _randn((token_count, hidden_size), seed=34)

        reference_inputs = [
            value.detach().clone().requires_grad_(True)
            for value in (rmsnorm_output, gate_weight, up_weight, down_weight)
        ]
        reference_output, _, _, _ = _reference(*reference_inputs)
        reference_output.backward(grad_output)

        local_gate_weight = gate_weight[shard_start:shard_end].contiguous()
        local_up_weight = up_weight[shard_start:shard_end].contiguous()
        local_down_weight = down_weight[:, shard_start:shard_end].contiguous()
        local_gate = rmsnorm_output @ local_gate_weight.t()
        local_up = rmsnorm_output @ local_up_weight.t()
        local_activated = F.silu(local_gate) * local_up

        actual_grads = qwen3_ffn_backward(
            grad_output,
            rmsnorm_output,
            local_gate,
            local_up,
            local_activated,
            local_gate_weight,
            local_up_weight,
            local_down_weight,
            tp_group=dist.group.WORLD,
        )

        expected_grads = (
            reference_inputs[0].grad,
            reference_inputs[1].grad[shard_start:shard_end],
            reference_inputs[2].grad[shard_start:shard_end],
            reference_inputs[3].grad[:, shard_start:shard_end],
        )
        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "max_errors": [
                    float((actual - expected).abs().max().item())
                    for actual, expected in zip(actual_grads, expected_grads, strict=True)
                ],
            }
        )
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_qwen3_8b_dimensions_are_pinned():
    assert QWEN3_8B_HIDDEN_SIZE == 4096
    assert QWEN3_8B_INTERMEDIATE_SIZE == 12288


def test_backward_matches_autograd_reference(monkeypatch):
    stub = _TorchKernelStub()
    monkeypatch.setattr(ffn_module, "_C", stub)
    monkeypatch.setattr(ffn_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(ffn_module, "_validate_ffn_backward_inputs", lambda *args: None)

    hidden = _randn((2, 3, 8), seed=0)
    gate_weight = _randn((12, 8), seed=1)
    up_weight = _randn((12, 8), seed=2)
    down_weight = _randn((8, 12), seed=3)
    grad_output = _randn(hidden.shape, seed=4)

    ref_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected, gate, up, activated = _reference(*ref_inputs)
    expected.backward(grad_output)

    actual_grads = qwen3_ffn_backward(
        grad_output,
        hidden,
        gate.detach(),
        up.detach(),
        activated.detach(),
        gate_weight,
        up_weight,
        down_weight,
    )

    for actual, reference in zip(actual_grads, ref_inputs, strict=True):
        torch.testing.assert_close(actual, reference.grad)

    assert stub.calls.count("det_gemm_fwd") == 3
    assert stub.calls.count("det_gemm_db") == 3
    assert stub.calls.count("swiglu_backward") == 1


@requires_gloo
def test_tensor_parallel_backward_matches_full_reference_cpu_gloo_2_ranks():
    ctx = mp.get_context("spawn")
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "gloo_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_tp_ffn_backward_gloo_worker,
                args=(rank, world_size, init_method, result_queue),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()

        results = []
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=45))
        except queue.Empty:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail("timed out waiting for tensor-parallel Gloo workers")
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()

    for result in sorted(results, key=lambda item: item["rank"]):
        assert result["ok"], result.get("traceback")
        assert max(result["max_errors"]) < 1e-6
    for process in processes:
        assert process.exitcode == 0


def test_rejects_non_huggingface_weight_layout():
    hidden = torch.empty((2, 8), dtype=torch.bfloat16)
    grad_output = torch.empty_like(hidden)
    gate = torch.empty((2, 12), dtype=torch.bfloat16)
    up = torch.empty_like(gate)
    activated = torch.empty_like(gate)
    gate_weight = torch.empty((8, 12), dtype=torch.bfloat16)
    up_weight = torch.empty((12, 8), dtype=torch.bfloat16)
    down_weight = torch.empty((8, 12), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="gate_weight must have shape"):
        qwen3_ffn_backward(
            grad_output,
            hidden,
            gate,
            up,
            activated,
            gate_weight,
            up_weight,
            down_weight,
        )


@requires_cuda_ffn
def test_cuda_forward_backward_matches_fp32_reference():
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

    hidden_2d = hidden.reshape(-1, hidden.size(-1))
    gate = _C.det_gemm_fwd(hidden_2d, gate_weight.t().contiguous())
    up = _C.det_gemm_fwd(hidden_2d, up_weight.t().contiguous())
    activated = _C.swiglu_forward(gate, up)
    actual_grads = qwen3_ffn_backward(
        grad_output,
        hidden,
        gate.reshape(*hidden.shape[:-1], gate.size(-1)),
        up.reshape(*hidden.shape[:-1], up.size(-1)),
        activated.reshape(*hidden.shape[:-1], activated.size(-1)),
        gate_weight,
        up_weight,
        down_weight,
    )

    for actual, reference in zip(actual_grads, ref_inputs, strict=True):
        torch.testing.assert_close(actual.cpu().float(), reference.grad, atol=5e-2, rtol=2e-2)


@requires_cuda_ffn
def test_cuda_forward_and_input_gradient_are_batch_invariant():
    gate_weight = _randn((128, 64), seed=20, device="cuda", dtype=torch.bfloat16)
    up_weight = _randn((128, 64), seed=21, device="cuda", dtype=torch.bfloat16)
    down_weight = _randn((64, 128), seed=22, device="cuda", dtype=torch.bfloat16)
    hidden = _randn((6, 64), seed=23, device="cuda", dtype=torch.bfloat16)
    grad_output = _randn(hidden.shape, seed=24, device="cuda", dtype=torch.bfloat16)

    def _saved_forward(value):
        gate = _C.det_gemm_fwd(value, gate_weight.t().contiguous())
        up = _C.det_gemm_fwd(value, up_weight.t().contiguous())
        return gate, up, _C.swiglu_forward(gate, up)

    full_saved = _saved_forward(hidden)
    full_grad_hidden = qwen3_ffn_backward(
        grad_output,
        hidden,
        *full_saved,
        gate_weight,
        up_weight,
        down_weight,
    )[0]

    hidden_slice = hidden[2:4]
    slice_saved = _saved_forward(hidden_slice)
    slice_grad_hidden = qwen3_ffn_backward(
        grad_output[2:4],
        hidden_slice,
        *slice_saved,
        gate_weight,
        up_weight,
        down_weight,
    )[0]

    assert torch.equal(slice_grad_hidden, full_grad_hidden[2:4])
