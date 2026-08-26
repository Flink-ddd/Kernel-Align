# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Qwen3 module correspondence and call chain.

Every ``equivalence`` carries a ``verified_by``: unproven, "filtering false
positives" quietly becomes "hiding real findings".
"""

from __future__ import annotations

from rl_engine.mismatch.schema import ModuleCorrespondence, PropagationEdge

QWEN3_CORRESPONDENCES: tuple[ModuleCorrespondence, ...] = (
    ModuleCorrespondence(
        semantic_name="attn.qkv",
        training_module="megatron.core.transformer.attention.linear_qkv",
        rollout_module="vllm.model_executor.models.qwen3.qkv_proj",
        equivalence="concat_on_dim0",
        verified_by="tests/test_mismatch_model_meta.py::test_qwen3_qkv_concat_equivalence",
    ),
    ModuleCorrespondence(
        semantic_name="attn.out",
        training_module="megatron.core.transformer.attention.linear_proj",
        rollout_module="vllm.model_executor.models.qwen3.o_proj",
    ),
    ModuleCorrespondence(
        semantic_name="mlp.gate_up",
        training_module="megatron.core.transformer.mlp.linear_fc1",
        rollout_module="vllm.model_executor.models.qwen3.gate_up_proj",
        equivalence="concat_on_dim0",
        verified_by="tests/test_mismatch_model_meta.py::test_qwen3_gate_up_concat_equivalence",
    ),
    ModuleCorrespondence(
        semantic_name="mlp.down",
        training_module="megatron.core.transformer.mlp.linear_fc2",
        rollout_module="vllm.model_executor.models.qwen3.down_proj",
    ),
    ModuleCorrespondence(
        semantic_name="norm.input",
        training_module="megatron.core.transformer.transformer_layer.input_layernorm",
        rollout_module="vllm.model_executor.models.qwen3.input_layernorm",
    ),
    ModuleCorrespondence(
        semantic_name="lm_head",
        training_module="megatron.core.models.gpt.gpt_model.output_layer",
        rollout_module="vllm.model_executor.models.qwen3.lm_head",
    ),
)


QWEN3_EDGES: tuple[PropagationEdge, ...] = (
    PropagationEdge(upstream="norm.input", downstream="attn.qkv"),
    PropagationEdge(upstream="attn.qkv", downstream="attn.out"),
    PropagationEdge(upstream="attn.out", downstream="mlp.gate_up"),
    PropagationEdge(upstream="mlp.gate_up", downstream="mlp.down"),
    PropagationEdge(upstream="mlp.down", downstream="lm_head"),
)


QWEN3_8B_SHAPE = "L=36,H=4096,Hq=32,Hkv=8,D=128"
QWEN3_0B5_SHAPE = "L=24,H=896,Hq=14,Hkv=2,D=64"
QWEN3_SINGLE_LAYER_SHAPE = "L=1,H=896,Hq=14,Hkv=2,D=64"


__all__ = [
    "QWEN3_0B5_SHAPE",
    "QWEN3_8B_SHAPE",
    "QWEN3_CORRESPONDENCES",
    "QWEN3_EDGES",
    "QWEN3_SINGLE_LAYER_SHAPE",
]
