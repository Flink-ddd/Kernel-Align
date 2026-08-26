# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import rl_engine.integrations.framework_operators as framework_operators
import torch
from rl_engine.integrations.ablation import (
    IntegrationPlan,
    configure_integration_environment,
    integration_plan_from_environment,
)
from rl_engine.integrations.framework_operators import (
    MegatronAttentionOperator,
    SemanticOperatorHandle,
    _megatron_zigzag_layout,
    _packed_local_sequence_layout,
    _vllm_kv_cache_views,
)
from rl_engine.integrations.megatron_runtime import install_megatron_integration
from rl_engine.integrations.runtime import FrameworkOperatorIntegration
from rl_engine.integrations.state import clear_active_integration
from rl_engine.integrations.vllm_runtime import configure_vllm_environment
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
from rl_engine.kernels.ops.cuda.attention.strict_runtime import StrictCUDAAttentionRuntime


def test_framework_adapters_do_not_construct_registered_kernels_directly():
    source_path = (
        Path(__file__).parents[1] / "rl_engine" / "integrations" / "framework_operators.py"
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
    monkeypatch.setattr(framework_operators, "_megatron_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(framework_operators, "_require_nvidia_cuda", lambda tensor, module: None)
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
    assert len(calls) == 2
    assert [call["contract"].sharding.global_sequence_length for call in calls] == [
        8,
        8,
    ]
    assert [call["query_position_ids"].tolist() for call in calls] == [
        [[0, 1, 6, 7]],
        [[0, 1, 6, 7]],
    ]


def test_vllm_current_flash_attention_kv_cache_layout_is_materialized():
    cache = torch.arange(2 * 3 * 4 * 10).reshape(2, 3, 4, 10)

    key, value = _vllm_kv_cache_views(cache, head_size=5)

    assert key.shape == (2, 4, 3, 5)
    assert value.shape == (2, 4, 3, 5)
    assert torch.equal(key, cache.transpose(1, 2)[..., :5])
    assert torch.equal(value, cache.transpose(1, 2)[..., 5:])


def _cp1_contract(tokens: int) -> AttentionContract:
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=tokens,
        head_dim=4,
        causal=True,
        causal_offsets=(0,),
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


def test_strict_cuda_runtime_pins_training_to_single_query_prefixes(monkeypatch):
    calls: list[tuple[int, int, bool]] = []

    class Core:
        core_id = STRICT_ATTENTION_PRODUCTION_CORE_ID
        strict_schedule = STRICT_ATTENTION_FA4_SCHEDULE_ID

        def forward_bshd_with_lse(self, q, k, v, *, causal, **kwargs):
            del v, kwargs
            calls.append((q.size(1), k.size(1), causal))
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros((q.size(0), q.size(2), q.size(1)), dtype=torch.float32),
                provenance={"actual_backend": "fake.cuda.fa4"},
            )

    monkeypatch.setattr(
        StrictCUDAAttentionRuntime, "_require_nvidia_cuda", lambda self, tensor: None
    )
    runtime = StrictCUDAAttentionRuntime(core=Core(), communication=object())
    q = torch.zeros(1, 2, 4, 4, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    positions = torch.arange(4).unsqueeze(0)

    result = runtime.forward_with_lse(
        q,
        k,
        k,
        contract=_cp1_contract(4),
        causal=True,
        scale=0.5,
        cp_world_size=1,
        query_position_ids=positions,
        key_position_ids=positions,
    )

    assert calls == [(1, 1, False), (1, 2, False), (1, 3, False), (1, 4, False)]
    assert result.provenance["query_schedule"] == "single_query_causal_prefix"


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
