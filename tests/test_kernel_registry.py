# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels import registry as registry_module
from rl_engine.kernels.registry import KernelRegistry, OpBackend


def test_rocm_attention_defaults_to_flash_attention(monkeypatch):
    monkeypatch.delenv("RL_KERNEL_ROCM_ATTN_BACKEND", raising=False)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.PYTORCH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]


@pytest.mark.parametrize("value", ["FLASH_ATTN", "flash-attn", "Flash_Attention", " flash_attn "])
def test_rocm_attention_flash_opt_in_aliases(monkeypatch, value):
    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", value)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.PYTORCH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]


@pytest.mark.parametrize("value", ["native", "PYTORCH", " sdpa "])
def test_rocm_attention_can_opt_out_to_sdpa(monkeypatch, value):
    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", value)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.PYTORCH_ATTN,
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]


def test_rocm_attention_env_override_wins_after_hardware_adjustment(monkeypatch):
    def fake_hardware_adjustment(registry):
        registry._priority_map["rocm"]["attn"] = [
            OpBackend.PYTORCH_ATTN,
            OpBackend.ROCM_FLASH_ATTN,
            OpBackend.TRITON_GENERIC,
        ]

    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", "flash_attn")
    monkeypatch.setattr(KernelRegistry, "_adjust_priority_for_hardware", fake_hardware_adjustment)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.PYTORCH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]


def test_rocm_attention_unknown_env_value_uses_default_and_warns(monkeypatch):
    warnings = []

    def fake_warning(message, *args):
        warnings.append(message % args)

    monkeypatch.setenv("RL_KERNEL_ROCM_ATTN_BACKEND", "unknown")
    monkeypatch.setattr(registry_module.logger, "warning", fake_warning)

    registry = KernelRegistry()

    assert registry._priority_map["rocm"]["attn"] == [
        OpBackend.ROCM_FLASH_ATTN,
        OpBackend.PYTORCH_ATTN,
        OpBackend.TRITON_GENERIC,
    ]
    assert any("Unknown RL_KERNEL_ROCM_ATTN_BACKEND=unknown" in warning for warning in warnings)


def test_sm90_linear_ops_prioritize_cuda_when_extension_symbols_exist(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    class FakeExtension:
        fused_linear_logp_sm90 = object()
        batch_invariant_linear_logp_sm90 = object()
        embedding_sm90_forward = object()
        lm_head_sm90_forward = object()

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(
        registry_module.torch.cuda,
        "get_device_capability",
        lambda device=None: (9, 0),
    )
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())

    registry = KernelRegistry()

    assert registry._priority_map["cuda"]["embedding"][0] is OpBackend.CUDA_SM90_EMBEDDING
    assert registry._priority_map["cuda"]["lm_head"][0] is OpBackend.CUDA_SM90_LM_HEAD
    assert registry._priority_map["cuda"]["batch_invariant_linear_logp"] == [
        OpBackend.CUDA_BATCH_INVARIANT_LINEAR_LOGP_SM90
    ]


def test_sm90_linear_ops_do_not_prioritize_cuda_on_non_hopper(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    class FakeExtension:
        fused_linear_logp_sm90 = object()
        batch_invariant_linear_logp_sm90 = object()
        embedding_sm90_forward = object()
        lm_head_sm90_forward = object()

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(
        registry_module.torch.cuda,
        "get_device_capability",
        lambda device=None: (8, 0),
    )
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())

    registry = KernelRegistry()

    assert OpBackend.CUDA_SM90_EMBEDDING not in registry._priority_map["cuda"]["embedding"]
    assert OpBackend.CUDA_SM90_LM_HEAD not in registry._priority_map["cuda"]["lm_head"]
    assert (
        OpBackend.CUDA_FUSED_LINEAR_LOGP_SM90 not in registry._priority_map["cuda"]["linear_logp"]
    )
    assert registry._priority_map["cuda"]["batch_invariant_linear_logp"] == [
        OpBackend.CUDA_BATCH_INVARIANT_LINEAR_LOGP_SM90
    ]

    with pytest.raises(RuntimeError, match="No functional backend"):
        registry.get_op("batch_invariant_linear_logp", device="cuda")


def test_batch_invariant_linear_logp_dispatch_uses_requested_cuda_device(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    class FakeExtension:
        batch_invariant_linear_logp_sm90 = object()

    class FakeOp:
        pass

    probed_devices = []

    def capability(device=None):
        resolved = torch.device("cuda:0" if device is None else device)
        probed_devices.append(resolved)
        return (9, 0) if resolved.index == 1 else (8, 0)

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(registry_module.torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())
    monkeypatch.setattr(KernelRegistry, "_load_backend", lambda self, backend: FakeOp)

    registry = KernelRegistry()
    probed_devices.clear()

    with pytest.raises(RuntimeError, match="No functional backend"):
        registry.get_op("batch_invariant_linear_logp", device="cuda:0")
    assert OpBackend.CUDA_BATCH_INVARIANT_LINEAR_LOGP_SM90.name not in registry._failed_backends

    hopper_op = registry.get_op("batch_invariant_linear_logp", device="cuda:1")
    assert isinstance(hopper_op, FakeOp)
    assert registry.get_op("batch_invariant_linear_logp", device="cuda:1") is hopper_op

    with pytest.raises(RuntimeError, match="No functional backend"):
        registry.get_op("batch_invariant_linear_logp", device="cuda:0")

    assert torch.device("cuda:0") in probed_devices
    assert torch.device("cuda:1") in probed_devices


def test_batch_invariant_linear_logp_explicit_device_survives_init_probe_failure(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    class FakeExtension:
        batch_invariant_linear_logp_sm90 = object()

    class FakeOp:
        pass

    def capability(device=None):
        if device is None:
            raise RuntimeError("current device is unavailable")
        return (9, 0) if torch.device(device).index == 1 else (8, 0)

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(registry_module.torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())
    monkeypatch.setattr(KernelRegistry, "_load_backend", lambda self, backend: FakeOp)

    registry = KernelRegistry()

    assert isinstance(
        registry.get_op("batch_invariant_linear_logp", device="cuda:1"),
        FakeOp,
    )


def test_batch_invariant_linear_logp_hopper_fails_closed_without_symbol(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(
        registry_module.torch.cuda,
        "get_device_capability",
        lambda device=None: (9, 0),
    )
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", object())

    registry = KernelRegistry()

    with pytest.raises(RuntimeError, match="No functional backend"):
        registry.get_op("batch_invariant_linear_logp", device="cuda:0")
