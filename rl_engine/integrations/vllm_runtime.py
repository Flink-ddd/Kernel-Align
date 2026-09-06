# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM plugin hooks installed without editing vLLM source files."""

from __future__ import annotations

import importlib
import os
import pathlib
import re
import sys
from types import MethodType
from typing import Any

import torch

from rl_engine.distributed.collectives import (
    DETERMINISTIC_ALL_REDUCE_OP,
    collective_for_group,
    deterministic_all_reduce_inplace,
    deterministic_all_reduce_staged,
    deterministic_staging_reserve,
)
from rl_engine.integrations.ablation import (
    Implementation,
    IntegrationPlan,
    configure_integration_environment,
    integration_plan_from_environment,
)
from rl_engine.integrations.framework_operators import (
    VllmAttentionOperator,
    VllmFFNOperator,
    VllmLogpOperator,
    _strict_attention_projection_op,
)
from rl_engine.integrations.linear_logp import (
    clear_rollout_linear_logp_context,
    publish_rollout_linear_logp_context,
)
from rl_engine.integrations.state import get_active_integration, set_active_integration
from rl_engine.integrations.vllm import VllmIntegration
from rl_engine.kernels.ops.pytorch.ffn.ffn import register_packed_inference_observer
from rl_engine.kernels.ops.pytorch.norm.rms_norm import strict_add_rms_norm, strict_rms_norm

_PATCH_MARKER = "__rl_kernel_original_forward__"
_STRICT_MODEL_PATCH_MARKER = "__rl_kernel_original_strict_model_init__"
_STRICT_PROJECTION_MARKER = "__rl_kernel_strict_attention_projection__"
_STRICT_FFN_INIT_MARKER = "__rl_kernel_original_strict_ffn_init__"
_STRICT_RMS_NORM_INIT_MARKER = "__rl_kernel_original_strict_rms_norm_init__"
_STRICT_ATTENTION_RMS_NORM_MARKER = "__rl_kernel_strict_attention_rms_norm__"
_STRICT_ROTARY_INIT_MARKER = "__rl_kernel_original_strict_rotary_init__"
_STRICT_ROCM_ROPE_PATCH_MARKER = "__rl_kernel_original_strict_rocm_rope_forward__"
_STRICT_LM_HEAD_LINEAR_PATCH_MARKER = "__rl_kernel_original_lm_head_linear_apply__"
_STRICT_O_PROJ_COLLECTIVE_MARKER = "__rl_kernel_o_proj_collective__"
_STRICT_ROW_PARALLEL_PATCH_MARKER = "__rl_kernel_original_row_parallel_forward__"
_STRICT_DIRECT_STAGING_MARKER = "__rl_kernel_direct_staging_active__"
_STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER = "__rl_kernel_original_layer_diagnostic_forward__"
_RLK_ATTENTION_BACKEND: type[Any] | None = None
_RLK_ATTENTION_IMPL: type[Any] | None = None
_RLK_ATTENTION_BUILDER: type[Any] | None = None
_RLK_O_PROJ_COLLECTIVE_BACKEND: str | None = None
_VLLM_LAYER_DIAGNOSTIC_BUFFER: dict[str, Any] | None = None
_VLLM_LAYER_DIAGNOSTIC_CALLS = 0
_VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER: int | None = None


def _alignment_diagnostics_enabled() -> bool:
    value = os.getenv(
        "RL_KERNEL_LAYER_ALIGNMENT_DIAGNOSTICS",
        os.getenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS", ""),
    )
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _diagnostic_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.getenv("RANK", "0"))


def _patch_qwen3_layer_alignment_diagnostics() -> None:
    """Record one bounded semantic decoder trace per vLLM model forward."""

    if not _alignment_diagnostics_enabled():
        return
    from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3DecoderLayer
    from rl_engine.kernels.ops.rocm.attention.strict_runtime import StrictRocmAttentionRuntime

    if hasattr(Qwen3DecoderLayer, _STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER):
        return
    original_init = Qwen3DecoderLayer.__init__
    original_forward = Qwen3DecoderLayer.forward
    original_attention_forward = Qwen3Attention.forward
    original_gather_paged_row = StrictRocmAttentionRuntime._gather_paged_row

    def init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        original_init(instance, *args, **kwargs)
        config = args[0] if args else kwargs.get("config")
        prefix = kwargs.get("prefix", args[3] if len(args) > 3 else "")
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(prefix))
        if match is None:
            raise RuntimeError(f"cannot recover Qwen3 decoder layer from prefix {prefix!r}")
        instance._rl_kernel_layer_diagnostic_index = int(match.group(1))
        instance._rl_kernel_layer_diagnostic_count = int(config.num_hidden_layers)

    def forward_wrapped(
        instance: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _VLLM_LAYER_DIAGNOSTIC_BUFFER
        global _VLLM_LAYER_DIAGNOSTIC_CALLS
        global _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER

        layer = int(instance._rl_kernel_layer_diagnostic_index)
        layer_count = int(instance._rl_kernel_layer_diagnostic_count)
        max_rows = int(os.getenv("RL_KERNEL_ALIGNMENT_MAX_ROWS", "64"))
        if layer == 0:
            semantic_input = hidden_states if residual is None else hidden_states + residual
            if semantic_input.ndim != 2:
                raise RuntimeError("vLLM layer diagnostics require [T,H] hidden states")
            _VLLM_LAYER_DIAGNOSTIC_BUFFER = (
                {
                    "positions": positions.detach(),
                    "input": semantic_input.detach(),
                    "outputs": [],
                }
                if int(semantic_input.size(0)) <= max_rows
                else None
            )

        _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER = layer
        try:
            result = original_forward(instance, positions, hidden_states, residual)
        finally:
            _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER = None
        output, output_residual = result
        buffer = _VLLM_LAYER_DIAGNOSTIC_BUFFER
        if buffer is None:
            return result
        semantic_output = output + output_residual
        buffer["outputs"].append(semantic_output.detach())
        if layer == 0:
            layer_zero = buffer.setdefault("layer0", {})
            layer_zero["attention_residual"] = output_residual.detach()
            layer_zero["mlp_norm"] = torch.nn.functional.rms_norm(
                output_residual,
                (output_residual.shape[-1],),
                instance.post_attention_layernorm.weight,
                instance.post_attention_layernorm.variance_epsilon,
            ).detach()
            layer_zero["mlp_output"] = output.detach()
        if layer != layer_count - 1:
            return result
        if len(buffer["outputs"]) != layer_count:
            raise RuntimeError(
                "vLLM layer diagnostics did not observe every decoder layer: "
                f"{len(buffer['outputs'])} != {layer_count}"
            )
        call_index = _VLLM_LAYER_DIAGNOSTIC_CALLS
        _VLLM_LAYER_DIAGNOSTIC_CALLS += 1
        rank = _diagnostic_rank()
        payload = {
            "schema_version": "rlkernel.layer_alignment_diagnostic.v1",
            "framework": "vllm",
            "pid": os.getpid(),
            "rank": rank,
            "call_index": call_index,
            "positions": buffer["positions"].cpu(),
            "input": buffer["input"].cpu(),
            "outputs": torch.stack(buffer["outputs"]).cpu(),
        }
        if "layer0" in buffer:
            payload["layer0"] = {
                name: value.cpu() for name, value in buffer["layer0"].items()
            }
        _VLLM_LAYER_DIAGNOSTIC_BUFFER = None
        root = os.getenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR", "").strip()
        if root:
            output_dir = pathlib.Path(root) / "layers"
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                payload,
                output_dir
                / (
                    f"vllm-pid{os.getpid()}-rank{rank:05d}-"
                    f"call{call_index:08d}.pt"
                ),
            )
        return result

    def attention_forward_wrapped(
        instance: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        buffer = _VLLM_LAYER_DIAGNOSTIC_BUFFER
        if buffer is None or _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER != 0:
            return original_attention_forward(instance, positions, hidden_states)
        qkv, _ = instance.qkv_proj(hidden_states)
        q, k, v = qkv.split([instance.q_size, instance.kv_size, instance.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // instance.head_dim, instance.head_dim)
        q_by_head = instance.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // instance.head_dim, instance.head_dim)
        k_by_head = instance.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = instance.rotary_emb(positions, q, k)
        attention_core = instance.attn(q, k, v)
        output, _ = instance.o_proj(attention_core)
        buffer.setdefault("layer0", {}).update({
            "attention_norm": hidden_states.detach(),
            "qkv": qkv.detach(),
            "query": q.view(*q.shape[:-1], -1, instance.head_dim).detach(),
            "key": k.view(*k.shape[:-1], -1, instance.head_dim).detach(),
            "value": v.view(*v.shape[:-1], -1, instance.head_dim).detach(),
            "attention_core": attention_core.detach(),
            "attention_output": output.detach(),
        })
        return output

    def gather_paged_row_wrapped(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_row: torch.Tensor,
        cached_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = original_gather_paged_row(
            k_cache,
            v_cache,
            page_row,
            cached_length,
        )
        buffer = _VLLM_LAYER_DIAGNOSTIC_BUFFER
        if buffer is not None and _VLLM_LAYER_DIAGNOSTIC_ACTIVE_LAYER == 0:
            buffer.setdefault("layer0", {}).update(
                {
                    "logical_key": key.detach(),
                    "logical_value": value.detach(),
                }
            )
        return key, value

    setattr(Qwen3DecoderLayer, _STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER, original_forward)
    Qwen3DecoderLayer.__init__ = init_wrapped
    Qwen3DecoderLayer.forward = forward_wrapped
    setattr(Qwen3Attention, _STRICT_LAYER_DIAGNOSTIC_PATCH_MARKER, original_attention_forward)
    Qwen3Attention.forward = attention_forward_wrapped
    StrictRocmAttentionRuntime._gather_paged_row = staticmethod(gather_paged_row_wrapped)


def _o_proj_collective_backend() -> str | None:
    return _RLK_O_PROJ_COLLECTIVE_BACKEND


def _is_worker_sampler_profile_batch(value: Any) -> bool:
    """Return true for vLLM's KV-cache profiling dummy sampler batch."""

    req_ids = getattr(value, "req_ids", None)
    is_padding = getattr(value, "is_padding", None)
    if not isinstance(req_ids, list) or not req_ids:
        return False
    if not all(isinstance(item, str) and item.startswith("req_") for item in req_ids):
        return False
    if is_padding is None or not hasattr(is_padding, "all"):
        return False
    try:
        return bool(is_padding.all().item())
    except (AttributeError, RuntimeError, TypeError):
        return False


def _is_v1_sampler_profile_batch(value: Any) -> bool:
    """Identify the no-logprob dummy batch used to size vLLM's KV cache."""

    if getattr(value, "max_num_logprobs", object()) is not None:
        return False
    output_token_ids = getattr(value, "output_token_ids", None)
    spec_token_ids = getattr(value, "spec_token_ids", None)
    if not (
        isinstance(output_token_ids, list)
        and output_token_ids
        and all(isinstance(ids, list) and not ids for ids in output_token_ids)
        and isinstance(spec_token_ids, list)
        and len(spec_token_ids) == len(output_token_ids)
        and all(isinstance(ids, list) and not ids for ids in spec_token_ids)
    ):
        return False
    return (
        bool(getattr(value, "no_penalties", False))
        and getattr(value, "prompt_token_ids", None) is None
        and getattr(value, "allowed_token_ids_mask", None) is None
        and getattr(value, "bad_words_token_ids", None) == {}
        and getattr(value, "generators", None) == {}
    )


def plan_from_environment() -> IntegrationPlan:
    """Compatibility alias for the shared process plan loader."""

    return integration_plan_from_environment()


def configure_vllm_environment(plan: IntegrationPlan, *, readback_dir: str | None = None) -> None:
    """Export the plan inherited by Vime's vLLM server subprocess."""

    os.environ["RL_KERNEL_VLLM_INTEGRATION"] = "1"
    configure_integration_environment(plan, readback_dir=readback_dir)
    if plan.implementation_for("attention", "rollout") is Implementation.RL_KERNEL:
        os.environ["VLLM_ATTENTION_BACKEND"] = (
            "ROCM_AITER_FA" if torch.version.hip is not None else "FLASH_ATTN"
        )
    if plan.implementation_for("logp", "rollout") is Implementation.RL_KERNEL:
        real_vocab = os.getenv("RL_KERNEL_VLLM_REAL_VOCAB_SIZE", "").strip()
        padded_vocab = os.getenv("RL_KERNEL_VLLM_PADDED_VOCAB_SIZE", "").strip()
        if not real_vocab or not padded_vocab:
            raise RuntimeError(
                "strict rollout linear_logp requires "
                "RL_KERNEL_VLLM_REAL_VOCAB_SIZE and "
                "RL_KERNEL_VLLM_PADDED_VOCAB_SIZE"
            )


def _patch_qwen_lm_head_padding() -> None:
    """Make vLLM's TP LM-head partition identical to Megatron's padded layout."""

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    if hasattr(ParallelLMHead, _PATCH_MARKER):
        return
    real_vocab = int(os.environ["RL_KERNEL_VLLM_REAL_VOCAB_SIZE"])
    padded_vocab = int(os.environ["RL_KERNEL_VLLM_PADDED_VOCAB_SIZE"])
    if padded_vocab <= real_vocab or padded_vocab % 2:
        raise RuntimeError(
            f"invalid strict rollout vocab contract: real={real_vocab}, padded={padded_vocab}"
        )
    padding_size = padded_vocab - real_vocab
    original = ParallelLMHead.__init__
    original_weight_loader = VocabParallelEmbedding.weight_loader

    def strict_weight_loader(instance: Any, param: Any, loaded_weight: torch.Tensor) -> None:
        strict_real_vocab = getattr(instance, "_rl_kernel_real_vocab_size", None)
        output_dim = getattr(param, "output_dim", None)
        packed_dim = getattr(param, "packed_dim", None)
        if (
            strict_real_vocab is not None
            and output_dim is not None
            and packed_dim is None
            and loaded_weight.shape[output_dim] == int(strict_real_vocab)
        ):
            start_idx = int(instance.shard_indices.org_vocab_start_index)
            shard_size = int(instance.shard_indices.org_vocab_end_index - start_idx)
            available = max(0, min(shard_size, int(strict_real_vocab) - start_idx))
            if available:
                loaded_slice = loaded_weight.narrow(output_dim, start_idx, available)
                param[:available].data.copy_(loaded_slice)
            if available < shard_size:
                param[available:shard_size].data.fill_(0)
            if shard_size < param.shape[0]:
                param[shard_size:].data.fill_(0)
            return
        original_weight_loader(instance, param, loaded_weight)

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        if args:
            args = (padded_vocab, *args[1:])
        else:
            kwargs["num_embeddings"] = padded_vocab
        # vLLM normally stores original-vocab and added-vocab rows as separate
        # segments. Strict R/R needs Megatron's single contiguous padded vocab.
        kwargs["org_num_embeddings"] = padded_vocab
        kwargs["padding_size"] = padding_size
        original(instance, *args, **kwargs)
        instance._rl_kernel_real_vocab_size = real_vocab
        if (
            int(getattr(instance, "org_vocab_size", -1)) != padded_vocab
            or int(getattr(instance, "num_embeddings_padded", -1)) != padded_vocab
        ):
            raise RuntimeError(
                "strict rollout LM-head padding did not produce the Megatron "
                f"padded vocab layout {padded_vocab}"
            )

    setattr(ParallelLMHead, _PATCH_MARKER, original)
    if not hasattr(VocabParallelEmbedding, "__rl_kernel_original_weight_loader__"):
        setattr(
            VocabParallelEmbedding,
            "__rl_kernel_original_weight_loader__",
            original_weight_loader,
        )
        setattr(VocabParallelEmbedding, "weight_loader", strict_weight_loader)
    setattr(ParallelLMHead, "__init__", wrapped)


def _patch_qwen_compute_logits(integration: VllmIntegration) -> None:
    """Publish the exact hidden and padded LM-head shard to the sampler."""

    from vllm.distributed import get_pp_group, get_tp_group
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    classes = tuple(dict.fromkeys((Qwen3ForCausalLM, Qwen2ForCausalLM)))
    installed: list[str] = []
    for cls in classes:
        if hasattr(cls, _PATCH_MARKER):
            continue
        original = cls.compute_logits  # type: ignore[attr-defined]

        def wrapped(
            instance: Any,
            hidden_states: Any,
            *,
            _original: Any = original,
        ) -> Any:
            lm_head = getattr(instance, "lm_head", None)
            if lm_head is None:
                raise RuntimeError("strict rollout linear_logp requires a Qwen LM-head")
            setattr(lm_head, _STRICT_PROJECTION_MARKER, "lm_head")
            logits = _original(instance, hidden_states)
            if not get_pp_group().is_last_rank:
                return logits
            shard_indices = getattr(lm_head, "shard_indices", None)
            weight = getattr(lm_head, "weight", None)
            if lm_head is None or shard_indices is None or not isinstance(weight, torch.Tensor):
                raise RuntimeError("strict rollout linear_logp requires a real Qwen LM-head shard")
            tp = get_tp_group()
            tp_world = int(tp.world_size)
            publish_rollout_linear_logp_context(
                hidden_states,
                weight,
                getattr(lm_head, "bias", None),
                tp_group=tp.device_group if tp_world > 1 else None,
                vocab_start_index=int(shard_indices.padded_org_vocab_start_index),
                global_vocab_size=int(lm_head.num_embeddings_padded),
                real_vocab_size=int(
                    getattr(lm_head, "_rl_kernel_real_vocab_size", lm_head.org_vocab_size)
                ),
            )
            return logits

        setattr(cls, _PATCH_MARKER, original)
        setattr(cls, "compute_logits", wrapped)
        installed.append(f"{cls.__module__}.{cls.__name__}.compute_logits")
    if installed:
        integration.record_installed_hook("logp", ",".join(installed))


def _patch_strict_lm_head_linear(
    *,
    linear_method_cls: type[Any] | None = None,
    det_gemm: Any | None = None,
) -> None:
    """Route vLLM's existing LM-head projection through RL-Kernel det_gemm."""

    if linear_method_cls is None:
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

        linear_method_cls = UnquantizedEmbeddingMethod
    if hasattr(linear_method_cls, _STRICT_LM_HEAD_LINEAR_PATCH_MARKER):
        return
    previous_apply = linear_method_cls.apply
    if det_gemm is None:
        from rl_engine.kernels.ops.matmul.det_gemm import DetGemmOp

        det_gemm = DetGemmOp()

    def wrapped(
        method: Any,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, _STRICT_PROJECTION_MARKER, None) != "lm_head":
            return previous_apply(method, layer, x, bias)
        x_2d = x.reshape(-1, x.shape[-1])
        output_2d = det_gemm.linear(x_2d, layer.weight)
        output = output_2d.reshape(*x.shape[:-1], layer.weight.size(0))
        return output if bias is None else output + bias

    setattr(
        linear_method_cls,
        _STRICT_LM_HEAD_LINEAR_PATCH_MARKER,
        previous_apply,
    )
    setattr(linear_method_cls, "apply", wrapped)


def _patch_strict_rocm_rotary_embedding(rotary_cls: type[Any]) -> None:
    """Bind vLLM Qwen3 RotaryEmbedding to the shared ROCm RoPE kernel."""

    if torch.version.hip is None or hasattr(rotary_cls, _STRICT_ROCM_ROPE_PATCH_MARKER):
        return
    from rl_engine.kernels.ops.rocm.rotary_embedding.rope import RocmDeterministicRoPEOp

    operator = RocmDeterministicRoPEOp()
    original = rotary_cls.forward_cuda

    def strict_forward_cuda(
        instance: Any,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        head_size = int(getattr(instance, "head_size", 0))
        rotary_dim = int(getattr(instance, "rotary_dim", head_size))
        if head_size <= 0 or rotary_dim != head_size:
            raise RuntimeError(
                "strict ROCm Qwen3 RoPE requires full-dimension rotation: "
                f"rotary_dim={rotary_dim}, head_size={head_size}"
            )
        if positions.ndim not in (1, 2) or positions.numel() != query.shape[0]:
            raise RuntimeError(
                "strict ROCm vLLM RoPE positions must align with flattened query rows: "
                f"positions={tuple(positions.shape)}, query={tuple(query.shape)}"
            )
        flat_positions = positions.reshape(-1).to(device=query.device, dtype=torch.int64)

        def head_major(value: torch.Tensor) -> torch.Tensor:
            if value.ndim != 2 or value.shape[1] % head_size:
                raise RuntimeError(
                    "strict ROCm vLLM RoPE expects flattened [tokens, heads*head_dim] tensors"
                )
            tokens = value.shape[0]
            if tokens != flat_positions.numel():
                raise RuntimeError("strict ROCm vLLM RoPE Q/K must share the position count")
            heads = value.shape[1] // head_size
            # The HIP kernel indexes one position table across rows. A
            # head-major view avoids duplicating the table for every head and
            # keeps the dispatch to one deterministic launch per Q/K tensor.
            return value.view(tokens, heads, head_size).permute(1, 0, 2).contiguous()

        def restore(rotated: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            return rotated.permute(1, 0, 2).reshape_as(reference).contiguous()

        query_head_major = head_major(query)
        if key is None:
            return restore(operator(query_head_major, flat_positions), query), None
        key_head_major = head_major(key)
        rotated_query, rotated_key = operator.forward_pair(
            query_head_major,
            key_head_major,
            flat_positions,
        )
        return restore(rotated_query, query), restore(rotated_key, key)

    strict_forward_cuda.__name__ = getattr(original, "__name__", "forward_cuda")
    setattr(rotary_cls, _STRICT_ROCM_ROPE_PATCH_MARKER, original)
    rotary_cls.forward_cuda = strict_forward_cuda


def _configure_strict_ffn_compilation(vllm_config: Any | None = None) -> None:
    """Keep the graph-safe TP reduction inside vLLM CUDA graphs."""

    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        raise RuntimeError("strict TP FFN initialization requires an active vLLM config")

    compilation = vllm_config.compilation_config
    splitting_ops = compilation.splitting_ops
    if splitting_ops is None:
        raise RuntimeError("vLLM splitting operators were not finalized before model init")
    # Older strict runtimes split this op because their host-owned sequence
    # value was frozen at graph capture. Sequence allocation is now device
    # owned, so remove stale entries and preserve the user's graph mode.
    splitting_ops[:] = [op for op in splitting_ops if op != DETERMINISTIC_ALL_REDUCE_OP]


def _patch_qwen_ffn(integration: VllmIntegration) -> None:
    from vllm.model_executor.models.qwen2 import Qwen2MLP

    operator: VllmFFNOperator | None = None
    if integration.plan.implementation_for("ffn", "rollout") is Implementation.RL_KERNEL:
        operator = VllmFFNOperator()
        integration.install_operator("ffn", operator)
    if hasattr(Qwen2MLP, _PATCH_MARKER):
        raise RuntimeError("vLLM Qwen2MLP is already RL-Kernel patched")
    original = Qwen2MLP.forward

    if operator is not None:
        if hasattr(Qwen2MLP, _STRICT_FFN_INIT_MARKER):
            raise RuntimeError("vLLM Qwen2MLP init is already RL-Kernel patched")
        original_init = Qwen2MLP.__init__
        compiled_evidence_armed = False

        def wrapped_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            nonlocal compiled_evidence_armed
            original_init(instance, *args, **kwargs)
            _handle, tp_world_size = operator.bind_packed_inference(instance)
            if not compiled_evidence_armed:
                execution_mode = (
                    "eager" if getattr(torch.version, "hip", None) is not None
                    else "compiled_cuda_graph"
                )
                register_packed_inference_observer(
                    lambda: integration.record_execution(
                        "ffn", operator, execution_mode=execution_mode
                    )
                )
                compiled_evidence_armed = True
            if tp_world_size > 1 and getattr(torch.version, "hip", None) is None:
                _configure_strict_ffn_compilation()

        setattr(Qwen2MLP, _STRICT_FFN_INIT_MARKER, original_init)
        setattr(Qwen2MLP, "__init__", wrapped_init)

    def wrapped(instance: Any, hidden_states: Any) -> Any:
        def native(_module: Any, value: Any) -> Any:
            return original(instance, value)

        return integration.execute("ffn", native, instance, hidden_states)

    setattr(Qwen2MLP, _PATCH_MARKER, original)
    setattr(Qwen2MLP, "forward", wrapped)
    integration.record_installed_hook("ffn", "vllm.model_executor.models.qwen2.Qwen2MLP.forward")


def _install_flash_attn_ops_compatibility() -> None:
    """Supply the legacy rotary import tree when an FA4-only package is installed."""

    try:
        importlib.import_module("flash_attn.ops.triton.rotary")
        return
    except (ImportError, OSError, RuntimeError):
        pass

    from vllm.vllm_flash_attn import ops as bundled_ops
    from vllm.vllm_flash_attn.ops import triton as bundled_triton
    from vllm.vllm_flash_attn.ops.triton import rotary as bundled_rotary

    sys.modules.setdefault("flash_attn.ops", bundled_ops)
    sys.modules.setdefault("flash_attn.ops.triton", bundled_triton)
    sys.modules.setdefault("flash_attn.ops.triton.rotary", bundled_rotary)


def _patch_qwen3_strict_model(
    *,
    rms_norm_cls: type[Any] | None = None,
    rotary_cls: type[Any] | None = None,
    linear_method_cls: type[Any] | None = None,
    attention_cls: type[Any] | None = None,
    row_parallel_cls: type[Any] | None = None,
    det_gemm: Any | None = None,
) -> None:
    """Align vLLM's RMSNorm and Attention projections with Megatron."""

    production_classes = rms_norm_cls is None or linear_method_cls is None or attention_cls is None
    if production_classes:
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        from vllm.model_executor.models.qwen3 import Qwen3Attention

        rms_norm_cls = RMSNorm
        rotary_cls = RotaryEmbedding
        linear_method_cls = UnquantizedLinearMethod
        attention_cls = Qwen3Attention
        row_parallel_cls = RowParallelLinear
    assert rms_norm_cls is not None
    assert linear_method_cls is not None
    assert attention_cls is not None
    if rotary_cls is not None:
        _patch_strict_rocm_rotary_embedding(rotary_cls)
    if det_gemm is None:
        det_gemm = _strict_attention_projection_op()

    attention_init = attention_cls.__init__
    unquantized_apply = linear_method_cls.apply
    original_rms_forward_native = rms_norm_cls.forward_native

    def deterministic_linear_apply(
        method: Any,
        layer: Any,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(layer, _STRICT_PROJECTION_MARKER):
            return unquantized_apply(method, layer, x, bias)
        x_2d = x.reshape(-1, x.shape[-1])
        linear = getattr(det_gemm, "linear", None)
        collective = getattr(layer, _STRICT_O_PROJ_COLLECTIVE_MARKER, None)
        direct_output = None
        if (
            torch.version.hip is None
            and collective is not None
            and bias is None
            and linear is not None
        ):
            direct_output = collective.direct_staging_view(
                (x_2d.size(0), layer.weight.shape[0]),
                dtype=x.dtype,
            )
            if direct_output is not None:
                deterministic_staging_reserve(
                    direct_output,
                    collective_handle=int(collective._handle),
                )
        output_2d = (
            linear(x_2d, layer.weight, out=direct_output)
            if linear is not None
            else det_gemm(x_2d, layer.weight.t().contiguous())
        )
        setattr(layer, _STRICT_DIRECT_STAGING_MARKER, direct_output is not None)
        output = output_2d.reshape(*x.shape[:-1], layer.weight.shape[0])
        return output if bias is None else output + bias

    def strict_attention_rms_norm_forward(
        instance: Any,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is not None:
            raise RuntimeError("strict Attention Q/K RMSNorm does not accept a residual")
        if instance.variance_size_override is not None:
            raise RuntimeError("strict Attention Q/K RMSNorm requires the full head dimension")
        weight = instance.weight.data if instance.has_weight else None
        if weight is None:
            raise RuntimeError("strict Attention Q/K RMSNorm requires a weight")
        return strict_rms_norm(x, weight, eps=instance.variance_epsilon)

    def require_rocm_eager_runtime() -> None:
        if not production_classes or torch.version.hip is None:
            return
        from vllm.config import (
            CUDAGraphMode,
            CompilationMode,
            get_current_vllm_config_or_none,
        )

        config = get_current_vllm_config_or_none()
        model_config = None if config is None else config.model_config
        compilation_config = None if config is None else config.compilation_config
        if (
            model_config is None
            or model_config.enforce_eager is not True
            or compilation_config is None
            or compilation_config.mode != CompilationMode.NONE
            or compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            raise RuntimeError(
                "strict ROCm vLLM Attention requires enforce_eager with compilation "
                "and CUDA/HIP graph capture disabled"
            )

    def bind_attention_rms_norm(attention: Any, name: str) -> None:
        norm = getattr(attention, name, None)
        if not isinstance(norm, rms_norm_cls):
            raise RuntimeError(f"strict Qwen3 Attention requires an RMSNorm {name} instance")
        if getattr(norm, _STRICT_ATTENTION_RMS_NORM_MARKER, False):
            raise RuntimeError(f"strict Qwen3 Attention {name} is already bound")
        norm._forward_method = MethodType(strict_attention_rms_norm_forward, norm)
        setattr(norm, _STRICT_ATTENTION_RMS_NORM_MARKER, True)

    def strict_rms_norm_forward_cuda(
        instance: Any,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if instance.variance_size_override is not None:
            return original_rms_forward_native(instance, x, residual)
        weight = instance.weight.data if instance.has_weight else None
        if weight is None:
            return original_rms_forward_native(instance, x, residual)
        if residual is None:
            return strict_rms_norm(
                x,
                weight,
                eps=instance.variance_epsilon,
            )
        return strict_add_rms_norm(
            x,
            residual,
            weight,
            eps=instance.variance_epsilon,
        )

    def bind_o_proj_collective(module: Any) -> None:
        global _RLK_O_PROJ_COLLECTIVE_BACKEND

        if int(getattr(module, "tp_size", 1)) <= 1:
            _RLK_O_PROJ_COLLECTIVE_BACKEND = "none"
            return
        from vllm.distributed.parallel_state import get_tp_group

        coordinator = get_tp_group()
        group = getattr(coordinator, "device_group", coordinator)
        collective = collective_for_group(group)
        if collective is None:
            raise RuntimeError("strict rollout o_proj requires an initialized TP process group")
        setattr(module, _STRICT_O_PROJ_COLLECTIVE_MARKER, collective)
        backend_id = getattr(collective, "backend_id", None)
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise RuntimeError("strict rollout o_proj collective has no backend identity")
        _RLK_O_PROJ_COLLECTIVE_BACKEND = backend_id.strip()
        if torch.version.hip is not None:
            return
        max_capture = int(os.getenv("RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE", "0"))
        if max_capture <= 0:
            raise RuntimeError(
                "strict rollout direct staging requires a positive graph capture size"
            )
        collective.prepare_direct_staging_views(
            ((batch, int(module.weight.shape[0])) for batch in range(1, max_capture + 1)),
            dtype=module.weight.dtype,
        )

    if row_parallel_cls is not None and not hasattr(
        row_parallel_cls, _STRICT_ROW_PARALLEL_PATCH_MARKER
    ):
        row_parallel_forward = row_parallel_cls.forward

        def strict_row_parallel_forward(instance: Any, input_: torch.Tensor) -> Any:
            collective = getattr(instance, _STRICT_O_PROJ_COLLECTIVE_MARKER, None)
            if collective is None:
                return row_parallel_forward(instance, input_)

            if instance.input_is_parallel:
                input_parallel = input_
            else:
                from vllm.distributed import split_tensor_along_last_dim

                input_parallel = split_tensor_along_last_dim(
                    input_, num_partitions=instance.tp_size
                )[instance.tp_rank].contiguous()

            assert instance.quant_method is not None
            bias_ = None if (instance.tp_rank > 0 or instance.skip_bias_add) else instance.bias
            output_parallel = instance.quant_method.apply(instance, input_parallel, bias_)

            if instance.reduce_results and instance.tp_size > 1:
                if torch.version.hip is not None:
                    output = collective.all_reduce(output_parallel)
                elif bool(getattr(instance, _STRICT_DIRECT_STAGING_MARKER, False)):
                    output = deterministic_all_reduce_staged(
                        output_parallel,
                        collective_handle=int(collective._handle),
                    )
                else:
                    deterministic_all_reduce_inplace(
                        output_parallel,
                        collective_handle=int(collective._handle),
                    )
                    output = output_parallel
            else:
                output = output_parallel

            if not instance.return_bias:
                return output
            output_bias = instance.bias if instance.skip_bias_add else None
            return output, output_bias

        setattr(row_parallel_cls, _STRICT_ROW_PARALLEL_PATCH_MARKER, row_parallel_forward)
        row_parallel_cls.forward = strict_row_parallel_forward

    if not hasattr(rms_norm_cls, _STRICT_RMS_NORM_INIT_MARKER):
        rms_norm_init = rms_norm_cls.__init__

        def rms_norm_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
            rms_norm_init(instance, *args, **kwargs)
            instance._forward_method = instance.forward_cuda

        setattr(rms_norm_cls, _STRICT_RMS_NORM_INIT_MARKER, rms_norm_init)
        rms_norm_cls.__init__ = rms_norm_init_wrapped
        rms_norm_cls.forward_cuda = strict_rms_norm_forward_cuda

    if rotary_cls is not None and not hasattr(rotary_cls, _STRICT_ROTARY_INIT_MARKER):
        rotary_init = rotary_cls.__init__

        def rotary_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
            rotary_init(instance, *args, **kwargs)
            instance._forward_method = instance.forward_cuda

        setattr(rotary_cls, _STRICT_ROTARY_INIT_MARKER, rotary_init)
        rotary_cls.__init__ = rotary_init_wrapped

    if hasattr(attention_cls, _STRICT_MODEL_PATCH_MARKER):
        return

    def attention_init_wrapped(instance: Any, *args: Any, **kwargs: Any) -> None:
        require_rocm_eager_runtime()
        attention_init(instance, *args, **kwargs)
        setattr(instance.qkv_proj, _STRICT_PROJECTION_MARKER, "qkv")
        setattr(instance.o_proj, _STRICT_PROJECTION_MARKER, "o_proj")
        if torch.version.hip is not None and production_classes:
            bind_attention_rms_norm(instance, "q_norm")
            bind_attention_rms_norm(instance, "k_norm")
        bind_o_proj_collective(instance.o_proj)

    setattr(attention_cls, _STRICT_MODEL_PATCH_MARKER, attention_init)
    linear_method_cls.apply = deterministic_linear_apply
    attention_cls.__init__ = attention_init_wrapped


def _patch_sampler(integration: VllmIntegration, *, strict_linear_logp: bool) -> None:
    from vllm.v1.sample.sampler import Sampler

    if hasattr(Sampler, _PATCH_MARKER):
        raise RuntimeError("vLLM Sampler is already RL-Kernel patched")
    original = Sampler.forward
    if strict_linear_logp:
        operator = VllmLogpOperator(original, strict_linear_logp=True)
        integration.install_operator("logp", operator)

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        sampling_metadata = kwargs.get("sampling_metadata")
        if sampling_metadata is None and len(args) >= 2:
            sampling_metadata = args[1]
        if strict_linear_logp and _is_v1_sampler_profile_batch(sampling_metadata):
            # gpu_model_runner._dummy_sampler_run computes logits with a
            # synthetic no-logprob batch. It is an allocator warmup, not a
            # rollout scoring event, so do not route or count it as logp.
            try:
                return original(instance, *args, **kwargs)
            finally:
                clear_rollout_linear_logp_context()

        def native(_sampler: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            return original(instance, *call_args, **call_kwargs)

        return integration.execute("logp", native, instance, *args, **kwargs)

    setattr(Sampler, _PATCH_MARKER, original)
    setattr(Sampler, "forward", wrapped)
    integration.record_installed_hook("logp", "vllm.v1.sample.sampler.Sampler.forward")


def _patch_worker_sampler(integration: VllmIntegration, *, strict_linear_logp: bool) -> None:
    """Patch the CUDA worker sampler used by vLLM 0.27's V1 runner.

    vLLM has two sampler implementations: the graph-friendly
    ``vllm.v1.sample.Sampler`` and the CUDA worker sampler used by the
    production GPU runner. Qwen3 rollout on vLLM 0.27 uses the latter.
    """

    try:
        from vllm.v1.worker.gpu.sample.sampler import Sampler
    except ImportError:
        return

    if hasattr(Sampler, _PATCH_MARKER):
        raise RuntimeError("vLLM GPU worker Sampler is already RL-Kernel patched")
    original = Sampler.__call__
    operator = VllmLogpOperator(
        original,
        worker_sampler=True,
        strict_linear_logp=strict_linear_logp,
    )
    integration.install_operator("logp", operator)

    def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
        def native(_sampler: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            return original(instance, *call_args, **call_kwargs)

        if args and _is_worker_sampler_profile_batch(args[1] if len(args) > 1 else None):
            return original(instance, *args, **kwargs)
        return integration.execute("logp", native, instance, *args, **kwargs)

    setattr(Sampler, _PATCH_MARKER, original)
    setattr(Sampler, "__call__", wrapped)
    integration.record_installed_hook("logp", "vllm.v1.worker.gpu.sample.sampler.Sampler.__call__")


def _register_attention_backend(integration: VllmIntegration) -> None:
    global _RLK_ATTENTION_BACKEND, _RLK_ATTENTION_BUILDER, _RLK_ATTENTION_IMPL

    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    if torch.version.hip is not None:
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionBackend as PlatformAttentionBackend,
        )
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionImpl as PlatformAttentionImpl,
        )
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionMetadataBuilder as PlatformAttentionMetadataBuilder,
        )

        selected_backend = AttentionBackendEnum.ROCM_AITER_FA
    else:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend as PlatformAttentionBackend,
        )
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionImpl as PlatformAttentionImpl,
        )
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadataBuilder as PlatformAttentionMetadataBuilder,
        )

        selected_backend = AttentionBackendEnum.FLASH_ATTN

    operator: VllmAttentionOperator | None = None
    if integration.plan.implementation_for("attention", "rollout") is Implementation.RL_KERNEL:
        operator = VllmAttentionOperator(
            projection_collective_backend=_o_proj_collective_backend
        )
        integration.install_operator("attention", operator)

    class RlKernelAttentionImpl(PlatformAttentionImpl):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if operator is not None:
                operator.bind_inference()

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            active = get_active_integration("vllm")
            if active is not integration:
                raise RuntimeError("vLLM Attention executed without its installed integration")

            def native(_impl: Any, *call_args: Any, **call_kwargs: Any) -> Any:
                return PlatformAttentionImpl.forward(self, *call_args, **call_kwargs)

            return integration.execute("attention", native, self, *args, **kwargs)

    class RlKernelAttentionMetadataBuilder(PlatformAttentionMetadataBuilder):
        pass

    class RlKernelAttentionBackend(PlatformAttentionBackend):
        @staticmethod
        def get_impl_cls() -> type[Any]:
            return RlKernelAttentionImpl

        @staticmethod
        def get_builder_cls() -> type[Any]:
            return RlKernelAttentionMetadataBuilder

    RlKernelAttentionImpl.__module__ = __name__
    RlKernelAttentionImpl.__qualname__ = "RlKernelAttentionImpl"
    RlKernelAttentionMetadataBuilder.__module__ = __name__
    RlKernelAttentionMetadataBuilder.__qualname__ = "RlKernelAttentionMetadataBuilder"
    RlKernelAttentionBackend.__module__ = __name__
    RlKernelAttentionBackend.__qualname__ = "RlKernelAttentionBackend"
    _RLK_ATTENTION_IMPL = RlKernelAttentionImpl
    _RLK_ATTENTION_BUILDER = RlKernelAttentionMetadataBuilder
    _RLK_ATTENTION_BACKEND = RlKernelAttentionBackend
    globals()["RlKernelAttentionImpl"] = RlKernelAttentionImpl
    globals()["RlKernelAttentionMetadataBuilder"] = RlKernelAttentionMetadataBuilder
    globals()["RlKernelAttentionBackend"] = RlKernelAttentionBackend
    # Override the platform-selected enum so vLLM keeps its native metadata and
    # cache update path while RL-Kernel owns the attention arithmetic call.
    register_backend(
        selected_backend,
        f"{__name__}.RlKernelAttentionBackend",
    )
    integration.record_installed_hook("attention", f"{__name__}.RlKernelAttentionBackend")


def install_vllm_integration(plan: IntegrationPlan) -> VllmIntegration:
    """Install vLLM paged Attention, Qwen dense FFN and sampler Logp routes."""

    existing = get_active_integration("vllm")
    if existing is not None:
        if not isinstance(existing, VllmIntegration):
            raise RuntimeError("active vLLM integration has an unexpected type")
        if existing.plan != plan:
            raise RuntimeError("vLLM integration is already installed with another plan")
        return existing
    integration = VllmIntegration(plan, rl_kernel_operators={})
    set_active_integration("vllm", integration)
    # vLLM imports this legacy rotary path even when every operator is routed
    # to production. FA4-only installations need the bundled compatibility
    # namespace before any attention backend is imported.
    if torch.version.hip is None:
        _install_flash_attn_ops_compatibility()
    strict_linear_logp = plan.implementation_for("logp", "rollout") is Implementation.RL_KERNEL
    strict_attention = plan.implementation_for("attention", "rollout") is Implementation.RL_KERNEL
    if strict_attention:
        _patch_qwen3_strict_model()
    _patch_qwen3_layer_alignment_diagnostics()
    if strict_linear_logp:
        _patch_qwen_lm_head_padding()
        _patch_strict_lm_head_linear()
        _patch_qwen_compute_logits(integration)

    # Patch every boundary so P/R cases also produce production readback. The
    # integration object chooses native versus RL-Kernel per module.
    _register_attention_backend(integration)
    _patch_qwen_ffn(integration)
    if torch.version.hip is not None:
        # Current ROCm deployments use vLLM's V2 runner and its worker sampler.
        _patch_worker_sampler(integration, strict_linear_logp=strict_linear_logp)
    else:
        _patch_sampler(integration, strict_linear_logp=strict_linear_logp)
    if strict_linear_logp:
        sampler_hook = (
            "vllm.v1.worker.gpu.sample.sampler.Sampler.__call__"
            if torch.version.hip is not None
            else "vllm.v1.sample.sampler.Sampler.forward"
        )
        integration.record_installed_hook(
            "logp",
            "vllm.model_executor.models.qwen3.Qwen3ForCausalLM.compute_logits,"
            f"{sampler_hook}",
        )
    return integration


def register_vllm_plugin() -> None:
    """vLLM general-plugin entry point; inactive unless Vime exported a plan."""

    if os.getenv("RL_KERNEL_VLLM_INTEGRATION", "").strip() not in {"1", "true", "True"}:
        return
    install_vllm_integration(plan_from_environment())


__all__ = [
    "configure_vllm_environment",
    "install_vllm_integration",
    "plan_from_environment",
    "register_vllm_plugin",
]
