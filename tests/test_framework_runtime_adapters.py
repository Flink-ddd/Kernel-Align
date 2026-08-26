# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import ast
import os
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rl_engine.integrations.framework_operators as framework_operators
from rl_engine.distributed.collectives import DETERMINISTIC_ALL_REDUCE_OP
from rl_engine.integrations.ablation import (
    IntegrationPlan,
    configure_integration_environment,
    integration_plan_from_environment,
)
from rl_engine.integrations.framework_operators import (
    MegatronAttentionOperator,
    MegatronFFNOperator,
    SemanticOperatorHandle,
    VllmAttentionOperator,
    VllmFFNOperator,
    VllmLogpOperator,
    _megatron_zigzag_layout,
    _packed_local_sequence_layout,
    _vllm_kv_cache_views,
)
from rl_engine.integrations.megatron_runtime import (
    _patch_strict_attention_projections,
    _patch_strict_te_rms_norm,
    install_megatron_integration,
)
from rl_engine.integrations.runtime import FrameworkOperatorIntegration
from rl_engine.integrations.state import clear_active_integration
from rl_engine.integrations.vllm_runtime import (
    _configure_strict_ffn_compilation,
    _patch_qwen3_strict_model,
    _patch_strict_lm_head_linear,
    configure_vllm_environment,
)
from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    AttentionContract,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.ops.cuda.attention.strict_runtime import (
    StrictCUDAAttentionRuntime,
)
from rl_engine.kernels.ops.cuda.attention.flash_attn import (
    StrictFlashAttention4Core,
    StrictFlashAttentionUnavailable,
)


def test_framework_adapters_do_not_construct_registered_kernels_directly():
    source_path = (
        Path(__file__).parents[1]
        / "rl_engine"
        / "integrations"
        / "framework_operators.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {
        "AttentionAblationOp",
        "DeterministicCPAttentionReferenceOp",
        "Qwen3FFNOp",
        "StrictCUDAAttentionRuntime",
        "VocabParallelLogprobOp",
    }
    constructed = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert constructed.isdisjoint(forbidden)


def test_semantic_handle_uses_operator_bridge_and_exposes_instance_provenance():
    handle = SemanticOperatorHandle(
        target="training",
        semantic_op="attention",
        backend_id="rlkernel.attention.deterministic.v1",
    )
    tensor = torch.zeros(1, 1, 1, 8)

    first = handle.get(
        tensor,
        topology={
            "world_size": 1,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
    )
    second = handle.get(
        tensor,
        topology={
            "world_size": 1,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
    )

    assert first is second
    assert first.backend_id == "rlkernel.attention.deterministic.v1"
    assert handle.provenance is not None
    assert handle.provenance["semantic_op"] == "attention"
    assert handle.provenance["backend_id"] == first.backend_id


def test_plan_environment_is_shared_by_both_framework_installers(monkeypatch, tmp_path):
    for variable in (
        "RL_KERNEL_ATTENTION_CASE",
        "RL_KERNEL_FFN_CASE",
        "RL_KERNEL_LOGP_CASE",
        "RL_KERNEL_READBACK_DIR",
    ):
        monkeypatch.setenv(variable, "previous")
    plan = IntegrationPlan.from_case_ids(attention="P/R", ffn="R/P", logp="R/R")
    configure_integration_environment(plan, readback_dir=str(tmp_path))

    assert integration_plan_from_environment() == plan
    assert Path(os.environ["RL_KERNEL_READBACK_DIR"]) == tmp_path


def test_vllm_rlkernel_attention_overrides_selected_flash_attn_backend(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    plan = IntegrationPlan.from_case_ids(attention="P/R")

    configure_vllm_environment(plan)

    assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"


def test_megatron_install_is_idempotent_in_one_actor():
    class Attention:
        def forward(self, value):
            return value

    class FFN:
        def forward(self, value):
            return value

    plan = IntegrationPlan.from_case_ids()
    clear_active_integration("megatron")
    try:
        first = install_megatron_integration(
            plan,
            attention_classes=(Attention,),
            ffn_classes=(FFN,),
        )
        second = install_megatron_integration(
            plan,
            attention_classes=(Attention,),
            ffn_classes=(FFN,),
        )
        assert first is second
    finally:
        clear_active_integration("megatron")


def test_megatron_zigzag_positions_preserve_global_cp_ownership():
    rank_zero = _megatron_zigzag_layout(4, cp_rank=0, cp_world_size=2)
    rank_one = _megatron_zigzag_layout(4, cp_rank=1, cp_world_size=2)

    assert rank_zero == ((0, 1, 6, 7), (0, 3), (0, 6), (0, 2, 4))
    assert rank_one == ((2, 3, 4, 5), (1, 2), (2, 4), (0, 2, 4))
    assert sorted(rank_zero[0] + rank_one[0]) == list(range(8))


def test_packed_layout_recovers_local_offsets_from_global_cu_seqlens():
    packed = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8, 16], dtype=torch.int32),
    )

    local_offsets, global_lengths = _packed_local_sequence_layout(
        packed,
        cp_world_size=2,
        local_query_tokens=8,
        local_kv_tokens=8,
    )

    assert local_offsets == (0, 4, 8)
    assert global_lengths == (8, 8)


def test_megatron_packed_attention_runs_each_sequence_in_thd_order(monkeypatch):
    calls: list[dict[str, object]] = []

    class Operator:
        def bind_cuda_runtime(self, *, process_group=None):
            assert process_group == "cp-group"

        def __call__(self, q, k, v, **kwargs):
            del k, v
            calls.append(kwargs)
            return SimpleNamespace(
                out=q.clone(),
                provenance={
                    "actual_backend": "rlkernel.cuda.attention.fa4_ag_rs.v1",
                    "core_rows": [{"actual_backend": "rlkernel.cuda.attention.fa4.v1"}],
                },
            )

    operator = Operator()

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            assert tensor.shape == (8, 2, 4)
            assert topology["context_parallel_size"] == 2
            return operator

    parallel_state = SimpleNamespace(
        get_context_parallel_world_size=lambda: 2,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: "cp-group",
    )
    monkeypatch.setattr(
        framework_operators, "_megatron_parallel_state", lambda: parallel_state
    )
    monkeypatch.setattr(
        framework_operators, "_require_nvidia_cuda", lambda tensor, module: None
    )
    packed = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8, 16], dtype=torch.int32),
    )
    query = torch.zeros(8, 2, 4, dtype=torch.bfloat16)
    key = torch.zeros(8, 1, 4, dtype=torch.bfloat16)

    output = MegatronAttentionOperator(handle=Handle())(
        SimpleNamespace(softmax_scale=0.5),
        query,
        key,
        key,
        None,
        packed_seq_params=packed,
        num_splits=1,
    )

    assert output.shape == (8, 8)
    assert len(calls) == 1
    assert [call["contract"].sharding.global_sequence_length for call in calls] == [
        8,
    ]
    assert calls[0]["contract"].batch_size == 2
    assert [call["query_position_ids"].tolist() for call in calls] == [
        [[0, 1, 6, 7], [0, 1, 6, 7]],
    ]


def test_megatron_ffn_keeps_strict_gate_up_projections_separate(monkeypatch):
    calls: list[dict[str, object]] = []

    class Operator:
        def __call__(self, hidden_states, gate, up, down, **kwargs):
            calls.append(
                {
                    "hidden_states": hidden_states,
                    "gate": gate,
                    "up": up,
                    "down": down,
                    "kwargs": kwargs,
                }
            )
            return hidden_states

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            assert tensor.shape == (2, 8)
            assert topology == {
                "world_size": 4,
                "tensor_parallel_size": 2,
                "context_parallel_size": 2,
            }
            return Operator()

    parallel_state = SimpleNamespace(
        get_context_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_context_parallel_group=lambda: "cp-group",
    )
    monkeypatch.setattr(
        framework_operators, "_megatron_parallel_state", lambda: parallel_state
    )
    monkeypatch.setattr(
        framework_operators, "_require_nvidia_cuda", lambda tensor, module: None
    )
    fused_gate_up = torch.randn(12, 8)
    module = SimpleNamespace(
        config=SimpleNamespace(
            add_bias_linear=False,
            gated_linear_unit=True,
            sequence_parallel=False,
        ),
        linear_fc1=SimpleNamespace(weight=fused_gate_up),
        linear_fc2=SimpleNamespace(weight=torch.randn(8, 6)),
        tp_group="tp-group",
    )

    output, bias = MegatronFFNOperator(handle=Handle())(module, torch.randn(2, 8))

    assert output.shape == (2, 8)
    assert bias is None
    assert len(calls) == 1
    assert "fused_gate_up_weight" not in calls[0]["kwargs"]
    assert calls[0]["kwargs"]["tp_group"] == "tp-group"
    assert calls[0]["kwargs"]["cp_group"] == "cp-group"


def test_megatron_ffn_recovers_transformer_engine_fused_rms_norm(monkeypatch):
    calls: list[torch.Tensor] = []

    class Operator:
        def __call__(self, hidden_states, gate, up, down, **kwargs):
            del gate, up, down, kwargs
            calls.append(hidden_states)
            return hidden_states

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            del tensor, topology
            return Operator()

    parallel_state = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    monkeypatch.setattr(
        framework_operators, "_megatron_parallel_state", lambda: parallel_state
    )
    monkeypatch.setattr(
        framework_operators, "_require_nvidia_cuda", lambda tensor, module: None
    )
    norm_weight = torch.tensor([1.5, 0.5])
    linear_fc1 = SimpleNamespace(
        weight=torch.randn(4, 2),
        layer_norm_weight=norm_weight,
        layer_norm_bias=None,
        normalization="RMSNorm",
        zero_centered_gamma=False,
        eps=1e-6,
    )
    module = SimpleNamespace(
        config=SimpleNamespace(
            add_bias_linear=False,
            gated_linear_unit=True,
            sequence_parallel=False,
        ),
        linear_fc1=linear_fc1,
        linear_fc2=SimpleNamespace(weight=torch.randn(2, 2)),
        tp_group=None,
    )
    hidden_states = torch.tensor([[1.0, 2.0]])

    output, _ = MegatronFFNOperator(handle=Handle())(module, hidden_states)

    expected = torch.nn.functional.rms_norm(
        hidden_states,
        (2,),
        norm_weight,
        1e-6,
    )
    assert torch.equal(output, expected)
    assert torch.equal(calls[0], expected)


def test_vllm_ffn_uses_existing_packed_gate_up_weight(monkeypatch):
    calls: list[dict[str, object]] = []

    class Operator:
        def __call__(self, hidden_states, gate, up, down, **kwargs):
            calls.append(
                {
                    "hidden_states": hidden_states,
                    "gate": gate,
                    "up": up,
                    "down": down,
                    "kwargs": kwargs,
                }
            )
            return hidden_states

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            assert topology == {
                "world_size": 2,
                "tensor_parallel_size": 2,
                "context_parallel_size": 1,
            }
            return Operator()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        framework_operators, "_require_nvidia_cuda", lambda tensor, module: None
    )
    monkeypatch.setattr(
        framework_operators,
        "_vllm_tp_coordinates",
        lambda: (2, 0, "tp-group"),
    )
    fused_gate_up = torch.randn(12, 8)
    module = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=fused_gate_up),
        down_proj=SimpleNamespace(weight=torch.randn(8, 6)),
    )

    output = VllmFFNOperator(handle=Handle())(module, torch.randn(2, 8))

    assert output.shape == (2, 8)
    assert len(calls) == 1
    assert calls[0]["kwargs"]["fused_gate_up_weight"] is fused_gate_up
    assert calls[0]["kwargs"]["tp_group"] == "tp-group"


def test_vllm_ffn_compiled_path_uses_prebound_collective(monkeypatch):
    calls: list[dict[str, object]] = []
    coordinate_calls = 0

    def tp_coordinates():
        nonlocal coordinate_calls
        coordinate_calls += 1
        return 2, 0, "tp-group"

    class Operator:
        def prepare_packed_inference(self, fused, down, *, tp_group):
            calls.append(
                {
                    "phase": "prepare",
                    "fused": fused,
                    "down": down,
                    "tp_group": tp_group,
                }
            )
            return 1234, 2

        def packed_inference(
            self,
            hidden_states,
            fused,
            down,
            *,
            collective_handle,
            tp_world_size,
        ):
            calls.append(
                {
                    "phase": "execute",
                    "fused": fused,
                    "down": down,
                    "collective_handle": collective_handle,
                    "tp_world_size": tp_world_size,
                }
            )
            return hidden_states

    operator = Operator()

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            assert topology == {
                "world_size": 2,
                "tensor_parallel_size": 2,
                "context_parallel_size": 1,
            }
            return operator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch._dynamo, "is_compiling", lambda: True)
    monkeypatch.setattr(
        framework_operators, "_require_nvidia_cuda", lambda tensor, module: None
    )
    monkeypatch.setattr(
        framework_operators,
        "_vllm_tp_coordinates",
        tp_coordinates,
    )
    fused_gate_up = torch.randn(12, 8)
    down = torch.randn(8, 6)
    module = SimpleNamespace(
        gate_up_proj=SimpleNamespace(weight=fused_gate_up),
        down_proj=SimpleNamespace(weight=down),
    )
    adapter = VllmFFNOperator(handle=Handle())

    adapter.bind_packed_inference(module)
    assert adapter.provenance["execution"] == {
        "framework_layout": "vllm_tensor_parallel",
        "tp_world_size": 2,
        "runtime_platform": "cuda",
        "actual_backend": "rlkernel.cuda.det_gemm_swiglu",
        "gemm_backend": "rlkernel.det_gemm.sm90.v1",
        "gate_up_projection": "packed_single_launch",
        "triton_used": False,
    }
    output = adapter(module, torch.randn(2, 8))

    assert output.shape == (2, 8)
    assert coordinate_calls == 1
    assert calls == [
        {
            "phase": "prepare",
            "fused": fused_gate_up,
            "down": down,
            "tp_group": "tp-group",
        },
        {
            "phase": "execute",
            "fused": fused_gate_up,
            "down": down,
            "collective_handle": 1234,
            "tp_world_size": 2,
        },
    ]


def test_vllm_strict_tp_ffn_preserves_full_graph_for_device_sequenced_collective():
    class CudaGraphMode(Enum):
        NONE = 0
        PIECEWISE = 1
        FULL = 2
        FULL_AND_PIECEWISE = 3
        FULL_DECODE_ONLY = 4

    compilation = SimpleNamespace(
        splitting_ops=["vllm::unified_attention"],
        cudagraph_mode=CudaGraphMode.FULL_AND_PIECEWISE,
    )
    config = SimpleNamespace(compilation_config=compilation)

    _configure_strict_ffn_compilation(config)
    _configure_strict_ffn_compilation(config)

    assert compilation.cudagraph_mode is CudaGraphMode.FULL_AND_PIECEWISE
    assert compilation.splitting_ops == ["vllm::unified_attention"]


def test_vllm_attention_builds_paged_rows_from_device_metadata():
    operator = VllmAttentionOperator()
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        max_seq_len=4,
    )
    query = torch.zeros(2, 2, 4, dtype=torch.bfloat16)

    groups, summary = operator._materialization_groups(
        metadata,
        query=query,
        block_table=metadata.block_table,
        block_size=4,
        num_actual=2,
    )

    assert len(groups) == 1
    assert groups[0]["pages"].tolist() == [[0], [1]]
    assert groups[0]["query_indices"].tolist() == [0, 1]
    assert groups[0]["seqused_k"].tolist() == [4, 4]
    assert summary == {
        "row_count": 2,
        "query_position_range": "device_dynamic",
        "kv_token_range": "device_dynamic",
        "launch_group_count": 1,
        "metadata_source": "vllm_gpu",
        "metadata_reused_across_layers": False,
    }


def test_vllm_attention_binds_after_worker_device_selection(monkeypatch):
    calls = []
    empty_calls = []

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            calls.append((tensor.device, topology))
            return object()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        framework_operators,
        "_vllm_tp_coordinates",
        lambda: (2, 1, "tp-group"),
    )
    def fake_empty(*args, **kwargs):
        empty_calls.append((args, kwargs))
        return torch.zeros(1)

    monkeypatch.setattr(torch, "empty", fake_empty)
    operator = VllmAttentionOperator(handle=Handle())

    assert calls == []
    operator.bind_inference()
    operator.bind_inference()

    assert len(calls) == 1
    assert empty_calls[0][1]["device"] == torch.device("cuda", 3)
    assert calls[0][1] == {
        "world_size": 2,
        "tensor_parallel_size": 2,
        "context_parallel_size": 1,
    }


def test_vllm_attention_collapses_mixed_prefixes_into_one_paged_launch():
    operator = VllmAttentionOperator()
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([5, 3], dtype=torch.int32),
        block_table=torch.tensor([[2, 3], [7, -1]], dtype=torch.int32),
        max_seq_len=5,
    )
    query = torch.zeros(3, 2, 4, dtype=torch.bfloat16)

    groups, summary = operator._materialization_groups(
        metadata,
        query=query,
        block_table=metadata.block_table,
        block_size=4,
        num_actual=3,
    )

    assert len(groups) == 1
    assert groups[0]["query_indices"].tolist() == [0, 1, 2]
    assert groups[0]["seqused_k"].tolist() == [4, 5, 3]
    assert groups[0]["pages"].tolist() == [[2, 3], [2, 3], [7, -1]]
    assert summary["kv_token_range"] == "device_dynamic"
    assert summary["launch_group_count"] == 1


def test_vllm_attention_reuses_device_metadata_once_per_model_forward():
    operator = VllmAttentionOperator()
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([5, 3], dtype=torch.int32),
        block_table=torch.tensor([[2, 3], [7, -1]], dtype=torch.int32),
        max_seq_len=5,
    )
    query = torch.zeros(3, 2, 4, dtype=torch.bfloat16)
    first_layer = object()
    second_layer = object()

    first, first_summary = operator._materialization_groups(
        metadata,
        query=query,
        block_table=metadata.block_table,
        block_size=4,
        num_actual=3,
        cache_owner=first_layer,
    )
    second, second_summary = operator._materialization_groups(
        metadata,
        query=query,
        block_table=metadata.block_table,
        block_size=4,
        num_actual=3,
        cache_owner=second_layer,
    )

    assert second[0]["pages"].data_ptr() == first[0]["pages"].data_ptr()
    assert second[0]["seqused_k"].data_ptr() == first[0]["seqused_k"].data_ptr()
    assert first_summary["metadata_reused_across_layers"] is False
    assert second_summary["metadata_reused_across_layers"] is True

    next_forward, next_summary = operator._materialization_groups(
        metadata,
        query=query,
        block_table=metadata.block_table,
        block_size=4,
        num_actual=3,
        cache_owner=first_layer,
    )
    assert next_forward[0]["pages"].data_ptr() != first[0]["pages"].data_ptr()
    assert next_summary["metadata_reused_across_layers"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_attention_device_metadata_replays_in_cuda_graph():
    operator = VllmAttentionOperator()
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor(
            [0, 2, 3], device="cuda", dtype=torch.int32
        ),
        seq_lens=torch.tensor([5, 3], device="cuda", dtype=torch.int32),
        block_table=torch.tensor(
            [[2, 3], [7, 8]], device="cuda", dtype=torch.int32
        ),
        max_seq_len=8,
    )
    query = torch.zeros(3, 2, 4, device="cuda", dtype=torch.bfloat16)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        operator._materialization_groups(
            metadata,
            query=query,
            block_table=metadata.block_table,
            block_size=4,
            num_actual=3,
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    first_layer = object()
    second_layer = object()
    with torch.cuda.graph(graph):
        groups, _summary = operator._materialization_groups(
            metadata,
            query=query,
            block_table=metadata.block_table,
            block_size=4,
            num_actual=3,
            cache_owner=first_layer,
        )
        reused_groups, reused_summary = operator._materialization_groups(
            metadata,
            query=query,
            block_table=metadata.block_table,
            block_size=4,
            num_actual=3,
            cache_owner=second_layer,
        )
    pages = groups[0]["pages"]
    seqused_k = groups[0]["seqused_k"]
    assert reused_groups[0]["pages"].data_ptr() == pages.data_ptr()
    assert reused_summary["metadata_reused_across_layers"] is True

    metadata.query_start_loc.copy_(
        torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    )
    metadata.seq_lens.copy_(
        torch.tensor([4, 6], device="cuda", dtype=torch.int32)
    )
    metadata.block_table.copy_(
        torch.tensor([[10, 11], [20, 21]], device="cuda", dtype=torch.int32)
    )
    graph.replay()
    torch.cuda.synchronize()

    assert seqused_k.tolist() == [4, 5, 6]
    assert pages.tolist() == [[10, 11], [20, 21], [20, 21]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_attention_cuda_graph_masks_capacity_padding_rows():
    operator = VllmAttentionOperator()
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 8], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor([8], device="cuda", dtype=torch.int32),
        block_table=torch.tensor([[3, 4]], device="cuda", dtype=torch.int32),
        max_seq_len=8,
    )
    query = torch.zeros(8, 2, 4, device="cuda", dtype=torch.bfloat16)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        operator._materialization_groups(
            metadata,
            query=query,
            block_table=metadata.block_table,
            block_size=4,
            num_actual=8,
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        groups, _summary = operator._materialization_groups(
            metadata,
            query=query,
            block_table=metadata.block_table,
            block_size=4,
            num_actual=8,
            cache_owner=object(),
        )

    metadata.query_start_loc.copy_(
        torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    )
    metadata.seq_lens.copy_(torch.tensor([5], device="cuda", dtype=torch.int32))
    metadata.block_table.copy_(
        torch.tensor([[10, 11]], device="cuda", dtype=torch.int32)
    )
    graph.replay()
    torch.cuda.synchronize()

    assert groups[0]["seqused_k"].tolist() == [1, 2, 3, 4, 5, 1, 1, 1]
    assert groups[0]["pages"].tolist() == [[10, 11]] * 8


def test_vllm_attention_writes_contiguous_rows_directly_to_output(monkeypatch):
    calls: list[dict[str, object]] = []

    class Runtime:
        def forward_paged_with_lse(self, q, k, v, **kwargs):
            del k, v
            out = kwargs["out"]
            assert out is not None
            out.fill_(3)
            calls.append({"q": q, "out": out, "kwargs": kwargs})
            return SimpleNamespace(
                out=out,
                provenance={
                    "actual_backend": "rlkernel.cuda.attention.fa4_ag_rs.v1",
                    "fallback": False,
                },
            )

    class Operator:
        def bind_cuda_runtime(self):
            return Runtime()

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            del tensor
            assert topology["tensor_parallel_size"] == 2
            return Operator()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        framework_operators, "_require_nvidia_cuda", lambda tensor, module: None
    )
    monkeypatch.setattr(
        framework_operators,
        "_vllm_tp_coordinates",
        lambda: (2, 0, "tp-group"),
    )
    adapter = VllmAttentionOperator(handle=Handle())
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        num_actual_tokens=2,
        max_seq_len=4,
    )
    query = torch.zeros(2, 2, 4, dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 2, 4, 1, 4, dtype=torch.bfloat16)
    output = torch.full((2, 8), -1, dtype=torch.bfloat16)

    result = adapter(
        SimpleNamespace(head_size=4, num_heads=2, scale=0.5),
        None,
        query,
        query,
        query,
        kv_cache,
        metadata,
        output=output,
    )

    assert result.data_ptr() == output.data_ptr()
    assert torch.equal(result, torch.full_like(result, 3))
    assert len(calls) == 1
    assert calls[0]["q"].data_ptr() == query.data_ptr()
    assert calls[0]["out"].data_ptr() == output.data_ptr()
    assert adapter.provenance["execution"]["direct_output_buffer"] is True


def test_vllm_current_flash_attention_kv_cache_layout_is_materialized():
    cache = torch.arange(2 * 3 * 4 * 10).reshape(2, 3, 4, 10)

    key, value = _vllm_kv_cache_views(cache, head_size=5)

    assert key.shape == (2, 4, 3, 5)
    assert value.shape == (2, 4, 3, 5)
    assert torch.equal(key, cache.transpose(1, 2)[..., :5])
    assert torch.equal(value, cache.transpose(1, 2)[..., 5:])


def test_megatron_strict_attention_projections_install_without_debug_environment(
    monkeypatch,
):
    monkeypatch.delenv("RL_KERNEL_MODEL_DEBUG_DIR", raising=False)

    class ColumnLinear:
        def __init__(self):
            self.allreduce_dgrad = True

        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input.new_full((*input.shape[:-1], weight.shape[0]), -1)

    class RowLinear:
        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input.new_full((*input.shape[:-1], weight.shape[0]), -1)

    class SelfAttention:
        def __init__(self):
            self.linear_qkv = ColumnLinear()
            self.linear_proj = RowLinear()

    _patch_strict_attention_projections(
        self_attention_cls=SelfAttention,
        column_linear_cls=ColumnLinear,
        row_linear_cls=RowLinear,
        det_gemm=lambda lhs, rhs: lhs @ rhs,
    )
    attention = SelfAttention()
    value = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    qkv = attention.linear_qkv._forward_impl(value, weight, bias=None)
    projection = attention.linear_proj._forward_impl(value, weight, bias=None)
    native = ColumnLinear()._forward_impl(value, weight, bias=None)

    assert torch.equal(qkv, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(projection, qkv)
    assert torch.equal(native, torch.full((1, 3), -1.0))
    assert attention.linear_qkv.allreduce_dgrad is False


def test_megatron_strict_attention_projections_preserve_te_fused_norm_and_tp_mapping():
    events: list[str] = []

    class LocalLinear:
        def _forward_impl(self, input, weight, *args, **kwargs):
            del weight, args, kwargs
            return input

    class TELinear:
        def __init__(self, weight):
            self.weight = weight

        def __call__(self, value):
            return self.forward(value)

    class TEQKV(TELinear):
        def __init__(self, weight, norm_weight):
            super().__init__(weight)
            self.layer_norm_weight = norm_weight
            self.layer_norm_bias = None
            self.normalization = "RMSNorm"
            self.zero_centered_gamma = False
            self.eps = 1e-6

        def forward(self, value):
            return value.new_full((*value.shape[:-1], self.weight.shape[0]), -1), None

    class TERow(TELinear):
        def forward(self, value):
            return value.new_full((*value.shape[:-1], self.weight.shape[0]), -1), None

    qkv_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    norm_weight = torch.tensor([1.5, 0.5])
    projection_weight = torch.eye(3)

    class SelfAttention:
        def __init__(self):
            self.linear_qkv = TEQKV(qkv_weight, norm_weight)
            self.linear_proj = TERow(projection_weight)

    def copy_to_tp(value):
        events.append("copy")
        return value

    def reduce_from_tp(value):
        events.append("reduce")
        return value

    _patch_strict_attention_projections(
        self_attention_cls=SelfAttention,
        column_linear_cls=LocalLinear,
        row_linear_cls=LocalLinear,
        det_gemm=lambda lhs, rhs: lhs @ rhs,
        copy_to_tp=copy_to_tp,
        reduce_from_tp=reduce_from_tp,
    )
    attention = SelfAttention()
    value = torch.tensor([[1.0, 2.0]])

    qkv, qkv_bias = attention.linear_qkv(value)
    projection, projection_bias = attention.linear_proj(qkv)

    normalized = torch.nn.functional.rms_norm(value, (2,), norm_weight, 1e-6)
    assert torch.equal(qkv, normalized @ qkv_weight.t())
    assert torch.equal(projection, qkv)
    assert qkv_bias is None
    assert projection_bias is None
    assert events == ["copy", "reduce"]


def test_strict_te_rms_norm_uses_pytorch_arithmetic():
    class RMSNorm:
        def __init__(self):
            self.weight = torch.tensor([1.5, 0.5])
            self.eps = 1e-6
            self.zero_centered_gamma = False

        def forward(self, value):
            return value.new_full(value.shape, -1)

    _patch_strict_te_rms_norm(RMSNorm)
    module = RMSNorm()
    value = torch.tensor([[1.0, 2.0]])

    result = module.forward(value)

    expected = torch.nn.functional.rms_norm(value, (2,), module.weight, module.eps)
    assert torch.equal(result, expected)


def test_vllm_qwen3_strict_model_installs_without_debug_environment(monkeypatch):
    monkeypatch.delenv("RL_KERNEL_MODEL_DEBUG_DIR", raising=False)

    class RMSNorm:
        def __init__(self):
            self.variance_size_override = None
            self.has_weight = True
            self.hidden_size = 2
            self.weight = torch.tensor([1.5, 0.5])
            self.variance_epsilon = 1e-6
            self._forward_method = self.forward_native

        def forward_cuda(self, x, residual=None):
            del residual
            return x.new_full(x.shape, -1)

        def forward_native(self, x, residual=None):
            del residual
            return x.new_full(x.shape, -2)

    class RotaryEmbedding:
        def __init__(self):
            self._forward_method = self.forward_native

        def forward_cuda(self, x):
            return x + 1

        def forward_native(self, x):
            return x - 1

    class LinearLayer:
        def __init__(self):
            self.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    class LinearMethod:
        def apply(self, layer, x, bias=None):
            del bias
            return x.new_full((*x.shape[:-1], layer.weight.shape[0]), -1)

    class Attention:
        def __init__(self):
            self.qkv_proj = LinearLayer()
            self.o_proj = LinearLayer()

    _patch_qwen3_strict_model(
        rms_norm_cls=RMSNorm,
        rotary_cls=RotaryEmbedding,
        linear_method_cls=LinearMethod,
        attention_cls=Attention,
        det_gemm=lambda lhs, rhs: lhs @ rhs,
    )
    attention = Attention()
    method = LinearMethod()
    value = torch.tensor([[1.0, 2.0]])
    norm = RMSNorm()
    rotary = RotaryEmbedding()

    assert torch.equal(
        method.apply(attention.qkv_proj, value),
        torch.tensor([[1.0, 2.0, 3.0]]),
    )
    assert torch.equal(
        method.apply(LinearLayer(), value),
        torch.full((1, 3), -1.0),
    )
    assert torch.equal(
        norm.forward_cuda(value),
        torch.nn.functional.rms_norm(value, (2,), norm.weight, 1e-6),
    )
    assert norm._forward_method == norm.forward_cuda
    assert rotary._forward_method == rotary.forward_cuda
    assert torch.equal(rotary._forward_method(value), value + 1)

    norm.variance_size_override = 1
    assert torch.equal(norm._forward_method(value), torch.full_like(value, -2))
    norm.variance_size_override = None
    norm.has_weight = False
    assert torch.equal(norm._forward_method(value), torch.full_like(value, -2))


def test_vllm_strict_lm_head_patch_routes_only_marked_projection():
    class LinearMethod:
        def apply(self, layer, x, bias=None):
            del bias
            return x.new_full((*x.shape[:-1], layer.weight.size(0)), -1)

    class Layer:
        def __init__(self, *, marked):
            self.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            if marked:
                setattr(self, "__rl_kernel_strict_attention_projection__", "lm_head")

    class DetGemm:
        @staticmethod
        def linear(x, weight):
            return x @ weight.t()

    _patch_strict_lm_head_linear(
        linear_method_cls=LinearMethod,
        det_gemm=DetGemm(),
    )
    method = LinearMethod()
    value = torch.tensor([[1.0, 2.0]])
    assert torch.equal(
        method.apply(Layer(marked=True), value),
        torch.tensor([[1.0, 2.0, 3.0]]),
    )
    assert torch.equal(
        method.apply(Layer(marked=False), value),
        torch.full((1, 3), -1.0),
    )


def test_vllm_logp_replaces_every_duplicate_sampled_token_column():
    logprobs_type = namedtuple(
        "LogprobsTensors",
        ("logprob_token_ids", "logprobs", "selected_token_ranks"),
    )

    @dataclass(frozen=True)
    class SamplerResult:
        sampled_token_ids: torch.Tensor
        logprobs_tensors: object

    operator = VllmLogpOperator(lambda *_args, **_kwargs: None)
    result = SamplerResult(
        sampled_token_ids=torch.tensor([[7], [8]]),
        logprobs_tensors=logprobs_type(
            logprob_token_ids=torch.tensor([[7, 7], [8, 9]]),
            logprobs=torch.tensor([[-0.1, -0.1], [-0.2, -0.3]]),
            selected_token_ranks=torch.tensor([1, 2]),
        ),
    )

    updated = operator._replace_sampled_value(
        result,
        token_ids=torch.tensor([7, 8]),
        selected=torch.tensor([-1.25, -2.5]),
        provenance={},
    )

    assert torch.equal(
        updated.logprobs_tensors.logprobs,
        torch.tensor([[-1.25, -1.25], [-2.5, -0.3]]),
    )
    assert operator.provenance["native_reference_compared"] is False
    assert operator.provenance["strict_selected_logp"]["shape"] == [2]
    assert "native_vs_rlkernel_selected_diff" not in operator.provenance


def test_vllm_strict_logp_preserves_raw_local_logits_before_native_sampling(monkeypatch):
    from rl_engine.integrations.linear_logp import publish_rollout_linear_logp_context

    logprobs_type = namedtuple(
        "LogprobsTensors",
        ("logprob_token_ids", "logprobs", "selected_token_ranks"),
    )

    @dataclass(frozen=True)
    class SamplerResult:
        sampled_token_ids: torch.Tensor
        logprobs_tensors: object

    captured = {}

    class LinearLogp:
        backend_id = "rlkernel.linear_logp.bitwise.v1"
        provenance = {"actual_backend": backend_id}

        @staticmethod
        def from_local_logits(local_logits, target_ids, **kwargs):
            captured["local_logits"] = local_logits.clone()
            captured["target_ids"] = target_ids.clone()
            captured["kwargs"] = kwargs
            return torch.tensor([-7.0])

    def native(_sampler, logits, _sampling_metadata, **_kwargs):
        logits.fill_(99)
        return SamplerResult(
            sampled_token_ids=torch.tensor([[4]]),
            logprobs_tensors=logprobs_type(
                logprob_token_ids=torch.tensor([[4]]),
                logprobs=torch.tensor([[-1.0]]),
                selected_token_ranks=torch.tensor([0]),
            ),
        )

    monkeypatch.setattr(framework_operators, "_require_nvidia_cuda", lambda *args: None)
    publish_rollout_linear_logp_context(
        torch.zeros(1, 2, dtype=torch.bfloat16),
        torch.zeros(3, 2, dtype=torch.bfloat16),
        None,
        tp_group=object(),
        vocab_start_index=3,
        global_vocab_size=6,
        real_vocab_size=5,
    )
    operator = VllmLogpOperator(native, strict_linear_logp=True)
    operator._linear_logp = LinearLogp()
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    result = operator(object(), logits, object())

    assert torch.equal(logits, torch.full_like(logits, 99))
    assert torch.equal(
        captured["local_logits"],
        torch.tensor([[3.0, 4.0, float("-inf")]], dtype=torch.bfloat16),
    )
    assert torch.equal(result.logprobs_tensors.logprobs, torch.tensor([[-7.0]]))


def _cp1_contract(tokens: int, *, batch_size: int = 1) -> AttentionContract:
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=batch_size,
        query_sequence_length=tokens,
        head_dim=4,
        causal=True,
        causal_offsets=(0,) * batch_size,
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            cp_rank=0,
            cp_world_size=1,
            global_q_heads=2,
            global_kv_heads=1,
            local_q_head_start=0,
            local_q_heads=2,
            local_kv_head_start=0,
            local_kv_heads=1,
            global_sequence_length=tokens,
            local_sequence_length=tokens,
            global_block_indices=(0,),
            global_block_token_starts=(0,),
            local_block_offsets=(0, tokens),
        ),
        reduction=ReductionSpec(),
    )


def test_strict_cuda_runtime_batches_training_sequence_in_one_causal_launch(monkeypatch):
    calls: list[tuple[int, int, int, bool]] = []

    class Core:
        core_id = STRICT_ATTENTION_PRODUCTION_CORE_ID
        strict_schedule = STRICT_ATTENTION_FA4_SCHEDULE_ID

        def forward_bshd_with_lse(self, q, k, v, *, causal, **kwargs):
            del v, kwargs
            calls.append((q.size(0), q.size(1), k.size(1), causal))
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros((q.size(0), q.size(2), q.size(1)), dtype=torch.float32),
                provenance={"actual_backend": "fake.cuda.fa4"},
            )

    monkeypatch.setattr(
        StrictCUDAAttentionRuntime, "_require_nvidia_cuda", lambda self, tensor: None
    )
    runtime = StrictCUDAAttentionRuntime(core=Core(), communication=object())
    q = torch.zeros(3, 2, 4, 4, dtype=torch.bfloat16)
    k = torch.zeros(3, 1, 4, 4, dtype=torch.bfloat16)
    positions = torch.arange(4).repeat(3, 1)

    result = runtime.forward_with_lse(
        q,
        k,
        k,
        contract=_cp1_contract(4, batch_size=3),
        causal=True,
        scale=0.5,
        cp_world_size=1,
        query_position_ids=positions,
        key_position_ids=positions,
    )

    assert calls == [(3, 4, 4, True)]
    assert result.provenance["query_schedule"] == (
        "full_sequence_causal_single_launch"
    )
    assert result.provenance["backward_schedule"] == (
        "fa4_deterministic_full_sequence"
    )
    assert result.provenance["core_row_count"] == 12
    assert result.provenance["core_launch_count"] == 1
    assert result.provenance["core_query_length"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fa4_full_sequence_preserves_prefix_bitwise_and_deterministic_backward():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FA4 strict schedule requires SM90")
    try:
        core = StrictFlashAttention4Core()
    except StrictFlashAttentionUnavailable as exc:
        pytest.skip(str(exc))

    torch.manual_seed(20260826)
    batch, tokens, q_heads, kv_heads, head_dim = 1, 16, 16, 4, 128
    base = (
        torch.randn(
            batch,
            tokens,
            q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.randn(
            batch,
            tokens,
            kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        torch.randn(
            batch,
            tokens,
            kv_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64).repeat(
        batch, 1
    )
    grad_output = torch.randn_like(base[0])

    prefix_outputs = []
    prefix_lses = []
    prefix_inputs = tuple(value.detach().clone().requires_grad_(True) for value in base)
    for query_index in range(tokens):
        prefix = core.forward_bshd_with_lse(
            prefix_inputs[0][:, query_index : query_index + 1],
            prefix_inputs[1][:, : query_index + 1],
            prefix_inputs[2][:, : query_index + 1],
            causal=False,
            scale=head_dim**-0.5,
            query_position_ids=positions[:, query_index : query_index + 1],
            key_position_ids=positions[:, : query_index + 1],
            output_dtype=torch.bfloat16,
        )
        prefix_outputs.append(prefix.out)
        prefix_lses.append(prefix.lse)
    prefix_out = torch.cat(prefix_outputs, dim=1)
    prefix_lse = torch.cat(prefix_lses, dim=2)
    prefix_grads = torch.autograd.grad(
        prefix_out, prefix_inputs, grad_output
    )

    full_runs = []
    for _ in range(2):
        full_inputs = tuple(value.detach().clone().requires_grad_(True) for value in base)
        full = core.forward_bshd_with_lse(
            *full_inputs,
            causal=True,
            scale=head_dim**-0.5,
            query_position_ids=positions,
            key_position_ids=positions,
            output_dtype=torch.bfloat16,
        )
        full_grads = torch.autograd.grad(full.out, full_inputs, grad_output)
        full_runs.append((full.out.detach(), full.lse.detach(), *full_grads))

    assert torch.equal(prefix_out, full_runs[0][0])
    assert torch.equal(prefix_lse, full_runs[0][1])
    assert torch.equal(prefix_grads[0], full_runs[0][2])
    assert all(
        torch.equal(first, second)
        for first, second in zip(full_runs[0], full_runs[1], strict=True)
    )


def test_strict_cuda_runtime_routes_paged_kv_without_dense_materialization(monkeypatch):
    calls = []

    class Core:
        core_id = STRICT_ATTENTION_PRODUCTION_CORE_ID
        strict_schedule = STRICT_ATTENTION_FA4_SCHEDULE_ID

        def forward_paged_bshd_with_lse(self, q, k, v, **kwargs):
            calls.append((q.shape, k.shape, v.shape, kwargs))
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros((q.size(0), q.size(2), q.size(1))),
                provenance={"attention_backend": "fake.cuda.fa4.paged"},
            )

    monkeypatch.setattr(
        StrictCUDAAttentionRuntime, "_require_nvidia_cuda", lambda self, tensor: None
    )
    runtime = StrictCUDAAttentionRuntime(core=Core(), communication=object())
    q = torch.zeros(3, 2, 1, 4, dtype=torch.bfloat16)
    cache = torch.zeros(4, 4, 1, 4, dtype=torch.bfloat16)
    pages = torch.tensor([[0], [1], [2]], dtype=torch.int32)
    lengths = torch.tensor([4, 3, 2], dtype=torch.int32)

    result = runtime.forward_paged_with_lse(
        q,
        cache,
        cache,
        page_table=pages,
        seqused_k=lengths,
        max_seqlen_k=4,
        scale=0.5,
    )

    assert len(calls) == 1
    assert calls[0][0] == torch.Size([3, 1, 2, 4])
    assert calls[0][3]["page_table"] is pages
    assert result.out.shape == q.shape
    assert result.provenance["query_schedule"] == "paged_single_query_batch"
    assert result.provenance["core_launch_count"] == 1
    assert result.provenance["fallback"] is False


class _ReadbackOperator:
    backend_id = "rlkernel.attention.test"

    def __init__(self, provenance):
        self.provenance = provenance

    def __call__(self, value):
        return value


@pytest.mark.parametrize(
    ("provenance", "match"),
    [
        ({"runtime_platform": "cpu"}, "non-cuda"),
        ({"runtime_platform": "cuda", "actual_backend": "triton.attention"}, "triton"),
        ({"runtime_platform": "cuda", "triton_used": True}, "triton"),
    ],
)
def test_strict_readback_rejects_non_cuda_and_triton(provenance, match):
    plan = IntegrationPlan.from_case_ids(attention="R/R")
    integration = FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=plan,
        rl_kernel_operators={"attention": _ReadbackOperator(provenance)},
    )
    integration.record_installed_hook("attention", "test.attention")
    integration.execute("attention", lambda value: value, "x")

    with pytest.raises(RuntimeError, match=match):
        integration.assert_strict_ready()


def test_strict_readback_accepts_cuda_without_triton():
    plan = IntegrationPlan.from_case_ids(attention="R/R")
    integration = FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=plan,
        rl_kernel_operators={
            "attention": _ReadbackOperator(
                {
                    "runtime_platform": "cuda",
                    "actual_backend": "rlkernel.cuda.fa4",
                    "triton_used": False,
                }
            )
        },
    )
    integration.record_installed_hook("attention", "test.attention")
    integration.execute("attention", lambda value: value, "x")

    integration.assert_strict_ready()


def test_compiled_custom_op_execution_is_valid_route_evidence():
    plan = IntegrationPlan.from_case_ids(ffn="R/R")
    operator = _ReadbackOperator(
        {
            "runtime_platform": "cuda",
            "actual_backend": "rlkernel.cuda.det_gemm_swiglu",
            "triton_used": False,
        }
    )
    integration = FrameworkOperatorIntegration(
        framework="vllm",
        target="rollout",
        plan=plan,
        rl_kernel_operators={"ffn": operator},
    )
    integration.record_installed_hook("ffn", "test.compiled_ffn")

    integration.record_execution(
        "ffn",
        operator,
        execution_mode="compiled_cuda_graph",
    )

    readback = integration.readback()["operators"]["ffn"]
    assert readback["call_count"] == 1
    assert readback["execution_mode"] == "compiled_cuda_graph"
    integration.assert_strict_ready()


def test_route_report_is_emitted_once(capsys, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    plan = IntegrationPlan.from_case_ids(attention="R/R")
    integration = FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=plan,
        rl_kernel_operators={
            "attention": _ReadbackOperator(
                {
                    "runtime_platform": "cuda",
                    "actual_backend": "rlkernel.cuda.fa4",
                    "fallback": False,
                }
            )
        },
    )

    integration.execute("attention", lambda value: value, "first")
    integration.execute("attention", lambda value: value, "second")

    lines = [line for line in capsys.readouterr().out.splitlines() if "[route]" in line]
    assert lines == [
        "[RL-Kernel][route] framework=megatron target=training module=attention "
        "requested=rl_kernel actual=rlkernel.cuda.fa4 fallback=false"
    ]


def test_strict_route_fails_when_provenance_reports_fallback(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_ROUTE_REPORT", "0")
    plan = IntegrationPlan.from_case_ids(ffn="R/R")
    integration = FrameworkOperatorIntegration(
        framework="vllm",
        target="rollout",
        plan=plan,
        rl_kernel_operators={
            "ffn": _ReadbackOperator(
                {
                    "runtime_platform": "cuda",
                    "actual_backend": "rlkernel.cuda.det_gemm_swiglu",
                    "fallback": True,
                }
            )
        },
    )

    with pytest.raises(RuntimeError, match="strict RL-Kernel route reported fallback"):
        integration.execute("ffn", lambda value: value, "x")
