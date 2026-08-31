# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tensor-layout adapters from framework boundaries to semantic operators.

This module owns no numerical kernel selection. Attention and FFN instances
are resolved by :class:`OperatorBridge`; Logp uses the existing contract-aware
``KernelRegistry`` dispatch shared with the Vime provider.
"""

from __future__ import annotations

import os
from dataclasses import replace
from threading import Lock
from typing import Any, Mapping, cast

import torch

from rl_engine.alignment.cross_config.operators import OperatorBridge, OperatorOverride
from rl_engine.integrations.linear_logp import LinearLogpWrapper, take_rollout_linear_logp_context
from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    AttentionContract,
    AttentionDType,
    AttentionMode,
    AttentionRole,
)
from rl_engine.kernels.attention_contract import ReductionSpec as AttentionReductionSpec
from rl_engine.kernels.attention_contract import ShardingSpec as AttentionShardingSpec
from rl_engine.kernels.attention_contract import SplitKVSpec
from rl_engine.kernels.logprob_contract import LogprobContract, LogprobDType, LogprobRole, MaskSpec
from rl_engine.kernels.logprob_contract import ReductionSpec as LogprobReductionSpec
from rl_engine.kernels.logprob_contract import ShardingSpec as LogprobShardingSpec
from rl_engine.kernels.ops.cuda.matmul.det_gemm import det_gemm_backend_id
from rl_engine.kernels.ops.pytorch.attention.ablation import AttentionAblationConfig
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import BACKEND_ID as LOGP_BACKEND_ID
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import DEFAULT_NUM_VOCAB_TILES
from rl_engine.kernels.semantic_registry import OperatorRequirements
from rl_engine.runtime_mode import strict_contract_enabled

ATTENTION_BACKEND_ID = "rlkernel.attention.deterministic.v1"
FFN_BACKEND_ID = "rlkernel.ffn.qwen3.deterministic.v1"


def _device_name(tensor: torch.Tensor) -> str:
    if tensor.device.type == "cuda" and torch.version.hip is not None:
        return "rocm"
    return tensor.device.type


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    """Describe a hot-path tensor without reading CUDA values on the host."""

    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }


def _tensor_debug_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    stats = _tensor_metadata(detached)
    if detached.numel() == 0:
        return stats
    values = detached.float()
    stats.update(
        {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
        }
    )
    return stats


def _diff_debug_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_mismatch": True,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    diff = (left.detach().float() - right.detach().float()).abs()
    stats = _tensor_debug_stats(diff)
    stats["mismatch_count"] = int(torch.ne(left.detach(), right.detach()).sum().item())
    return stats


def _attention_dtype(tensor: torch.Tensor) -> AttentionDType:
    try:
        return {
            torch.bfloat16: AttentionDType.BF16,
            torch.float16: AttentionDType.FP16,
            torch.float32: AttentionDType.FP32,
        }[tensor.dtype]
    except KeyError as exc:
        raise RuntimeError(f"unsupported Attention dtype {tensor.dtype}") from exc


def _require_nvidia_cuda(tensor: torch.Tensor, module: str) -> None:
    if tensor.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError(f"strict {module} R/R requires NVIDIA CUDA tensors")


class SemanticOperatorHandle:
    """Resolve one exact semantic backend once for one framework process."""

    def __init__(self, *, target: str, semantic_op: str, backend_id: str) -> None:
        if target not in {"training", "rollout"}:
            raise ValueError("target must be 'training' or 'rollout'")
        self.target = target
        self.semantic_op = semantic_op
        self.backend_id = backend_id
        self._bridge = OperatorBridge()
        self._instance: Any | None = None
        self._provenance: dict[str, Any] | None = None
        self._runtime_device: torch.device | None = None
        self._runtime_dtype: torch.dtype | None = None
        self._lock = Lock()

    def get(
        self,
        tensor: torch.Tensor,
        *,
        topology: Mapping[str, Any],
        factory_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        # This method is called from vLLM model forwards that may be captured
        # by torch.compile.  A Lock context manager is unsupported in a
        # Dynamo fullgraph, so keep the hot-path lookup lock-free.  Handles
        # are constructed per framework worker and resolution is idempotent;
        # duplicate first-call resolution is harmless and the bridge caches
        # the resulting semantic instance.
        if self._instance is not None:
            # vLLM constructs the plugin before its worker TP group exists, so
            # the eager prime may observe TP=1 while the first model call has
            # TP=2. The operator receives the live group at invocation time;
            # keep device and dtype strict, but do not reject this topology
            # transition after the semantic instance is resolved.
            if tensor.device != self._runtime_device or tensor.dtype != self._runtime_dtype:
                raise RuntimeError(
                    f"{self.semantic_op} runtime device/dtype changed after resolution"
                )
            return self._instance
        requirements = OperatorRequirements(
            device=_device_name(tensor),
            dtype=_dtype_name(tensor),
            topology=topology,
            alignment_properties={"deterministic": True},
        )
        target = cast(Any, self.target)
        resolved = self._bridge.resolve_override(
            OperatorOverride.for_target(
                semantic_op=self.semantic_op,
                backend_id=self.backend_id,
                target=target,
            ),
            requirements={self.target: requirements},
            strict=True,
        )
        instance = self._bridge.instantiate(
            resolved,
            target=target,
            factory_kwargs=factory_kwargs,
            cache=True,
        )
        provenance = self._bridge.instance_provenance(
            resolved,
            target=target,
            instance=instance,
        )
        actual_backend = getattr(instance, "backend_id", None)
        if actual_backend != self.backend_id:
            raise RuntimeError(
                f"semantic registry resolved {self.backend_id!r} but instantiated "
                f"{actual_backend!r}"
            )
        self._instance = instance
        self._provenance = provenance.to_dict()
        self._runtime_device = tensor.device
        self._runtime_dtype = tensor.dtype
        return instance

    @property
    def provenance(self) -> Mapping[str, Any] | None:
        return None if self._provenance is None else dict(self._provenance)


def _weight(module: Any, name: str) -> torch.Tensor:
    value = getattr(module, "weight", None)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{name} must expose an unquantized torch.Tensor weight")
    if value.ndim != 2:
        raise RuntimeError(f"{name}.weight must be two-dimensional")
    return value


def _fused_rms_norm_input(
    projection: Any,
    hidden_states: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Recover the RMSNorm hidden by TE's LayerNormLinear wrapper."""

    weight = getattr(projection, "layer_norm_weight", None)
    if weight is None:
        return hidden_states
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(f"{name}.layer_norm_weight must be a tensor")
    if getattr(projection, "normalization", None) != "RMSNorm":
        raise RuntimeError(f"strict {name} requires fused RMSNorm")
    if getattr(projection, "layer_norm_bias", None) is not None:
        raise RuntimeError(f"strict {name} RMSNorm must be bias-free")
    if bool(getattr(projection, "zero_centered_gamma", False)):
        raise RuntimeError(f"strict {name} does not support zero-centered gamma")
    eps = float(getattr(projection, "eps"))
    return torch.nn.functional.rms_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight,
        eps,
    )


def _split_gate_up(weight: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.size(0) % 2:
        raise RuntimeError(f"{name}.weight first dimension must contain equal gate/up shards")
    gate, up = weight.chunk(2, dim=0)
    return gate.contiguous(), up.contiguous()


def _megatron_parallel_state() -> Any:
    try:
        from megatron.core import parallel_state
    except ImportError as exc:  # pragma: no cover - exercised in framework environment
        raise RuntimeError("Megatron parallel_state is unavailable") from exc
    return parallel_state


def _megatron_zigzag_layout(
    local_tokens: int,
    *,
    cp_rank: int,
    cp_world_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if cp_world_size == 1:
        positions = tuple(range(local_tokens))
        return positions, (0,), (0,), (0, local_tokens)
    if local_tokens % 2:
        raise RuntimeError("Megatron zigzag CP requires an even local sequence length")
    chunk_size = local_tokens // 2
    second_index = 2 * cp_world_size - cp_rank - 1
    starts = (cp_rank * chunk_size, second_index * chunk_size)
    positions = tuple(range(starts[0], starts[0] + chunk_size)) + tuple(
        range(starts[1], starts[1] + chunk_size)
    )
    return positions, (cp_rank, second_index), starts, (0, chunk_size, local_tokens)


def _packed_local_sequence_layout(
    packed_seq_params: Any,
    *,
    cp_world_size: int,
    local_query_tokens: int,
    local_kv_tokens: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Recover local THD sequence offsets from Megatron's global cu_seqlens."""

    if str(getattr(packed_seq_params, "qkv_format", "")).lower() != "thd":
        raise RuntimeError("strict RL-Kernel packed Attention requires qkv_format='thd'")
    query_cu = getattr(packed_seq_params, "cu_seqlens_q", None)
    kv_cu = getattr(packed_seq_params, "cu_seqlens_kv", None)
    if not isinstance(query_cu, torch.Tensor) or not isinstance(kv_cu, torch.Tensor):
        raise RuntimeError("packed Attention requires tensor cu_seqlens_q/cu_seqlens_kv")
    query_offsets = tuple(
        int(value) for value in query_cu.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    kv_offsets = tuple(
        int(value) for value in kv_cu.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    if query_offsets != kv_offsets:
        raise RuntimeError("strict self-Attention requires identical Q and KV cu_seqlens")
    if len(query_offsets) < 2 or query_offsets[0] != 0:
        raise RuntimeError("packed Attention cu_seqlens must start at zero")
    global_lengths = tuple(
        right - left for left, right in zip(query_offsets[:-1], query_offsets[1:], strict=True)
    )
    if any(length <= 0 for length in global_lengths):
        raise RuntimeError("packed Attention cu_seqlens must be strictly increasing")
    if any(length % cp_world_size for length in global_lengths):
        raise RuntimeError("packed Attention sequence lengths must be divisible by CP size")
    local_lengths = tuple(length // cp_world_size for length in global_lengths)
    local_offsets = [0]
    for length in local_lengths:
        local_offsets.append(local_offsets[-1] + length)
    if local_offsets[-1] != local_query_tokens or local_offsets[-1] != local_kv_tokens:
        raise RuntimeError("packed Attention cu_seqlens do not cover the local Q/KV token rows")
    return tuple(local_offsets), global_lengths


def _tensor_cache_token(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Identify one tensor value while it is reused across framework layers."""

    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
        int(tensor._version),
    )


def _compact_attention_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep strict backend identity without retaining one record per token."""

    compact = dict(value)
    rows = compact.pop("core_rows", None)
    if isinstance(rows, (list, tuple)):
        compact["core_row_count"] = len(rows)
        backends = sorted(
            {
                str(row["actual_backend"])
                for row in rows
                if isinstance(row, Mapping) and row.get("actual_backend")
            }
        )
        compact["core_actual_backends"] = backends
    return compact


def _dense_attention_contract(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    role: AttentionRole,
    causal: bool,
    tp_rank: int,
    tp_world_size: int,
    cp_rank: int = 0,
    cp_world_size: int = 1,
    mode: AttentionMode | None = None,
    global_sequence_length: int | None = None,
    global_block_indices: tuple[int, ...] = (0,),
    global_block_token_starts: tuple[int, ...] = (0,),
    local_block_offsets: tuple[int, ...] | None = None,
) -> AttentionContract:
    batch, q_heads, query_tokens, head_dim = query.shape
    kv_heads = key.size(1)
    return AttentionContract(
        role=role,
        mode=mode or AttentionMode.PREFILL,
        dtype=_attention_dtype(query),
        batch_size=batch,
        query_sequence_length=query_tokens,
        head_dim=head_dim,
        causal=causal,
        causal_offsets=(0,) * batch if causal else None,
        sharding=AttentionShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            global_q_heads=q_heads * tp_world_size,
            global_kv_heads=kv_heads * tp_world_size,
            local_q_head_start=tp_rank * q_heads,
            local_q_heads=q_heads,
            local_kv_head_start=tp_rank * kv_heads,
            local_kv_heads=kv_heads,
            global_sequence_length=global_sequence_length or query_tokens,
            local_sequence_length=query_tokens,
            global_block_indices=global_block_indices,
            global_block_token_starts=global_block_token_starts,
            local_block_offsets=local_block_offsets or (0, query_tokens),
        ),
        reduction=AttentionReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        export_lse=True,
    )


class MegatronAttentionOperator:
    """Materialize Megatron layout, then call the registered Attention wrapper."""

    backend_id = ATTENTION_BACKEND_ID

    def __init__(self, handle: SemanticOperatorHandle | None = None) -> None:
        self._handle = handle or SemanticOperatorHandle(
            target="training", semantic_op="attention", backend_id=self.backend_id
        )
        self._last_provenance: dict[str, Any] = {}
        self._packed_layout_owner: Any | None = None
        self._packed_layout_key: tuple[Any, ...] | None = None
        self._packed_layout_value: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    def _packed_layout(
        self,
        packed_seq_params: Any,
        *,
        cp_world_size: int,
        local_query_tokens: int,
        local_kv_tokens: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        query_cu = getattr(packed_seq_params, "cu_seqlens_q", None)
        kv_cu = getattr(packed_seq_params, "cu_seqlens_kv", None)
        if not isinstance(query_cu, torch.Tensor) or not isinstance(kv_cu, torch.Tensor):
            raise RuntimeError("packed Attention requires tensor cu_seqlens_q/cu_seqlens_kv")
        key = (
            _tensor_cache_token(query_cu),
            _tensor_cache_token(kv_cu),
            cp_world_size,
            local_query_tokens,
            local_kv_tokens,
        )
        if self._packed_layout_owner is packed_seq_params and self._packed_layout_key == key:
            if self._packed_layout_value is None:
                raise RuntimeError("packed Attention layout cache is empty")
            return self._packed_layout_value
        value = _packed_local_sequence_layout(
            packed_seq_params,
            cp_world_size=cp_world_size,
            local_query_tokens=local_query_tokens,
            local_kv_tokens=local_kv_tokens,
        )
        self._packed_layout_owner = packed_seq_params
        self._packed_layout_key = key
        self._packed_layout_value = value
        return value

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "interface": "megatron.attention.forward",
            "operator": self.backend_id,
            "fallback": False,
            "semantic_instance": self._handle.provenance,
            "execution": dict(self._last_provenance),
        }

    def __call__(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        attn_mask_type: Any = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: Any = None,
        num_splits: int | None = None,
    ) -> torch.Tensor:
        del attn_mask_type
        if attention_bias is not None:
            raise RuntimeError("strict RL-Kernel Attention does not accept bias")
        if num_splits not in (None, 1):
            raise RuntimeError("strict RL-Kernel Attention requires num_splits=1")
        if attention_mask is not None and attention_mask.numel() > 1:
            raise RuntimeError("strict RL-Kernel Attention supports only its causal contract")
        expected_ndim = 3 if packed_seq_params is not None else 4
        if query.ndim != expected_ndim or key.ndim != expected_ndim or value.ndim != expected_ndim:
            layout = "[T, H, D]" if packed_seq_params is not None else "[S, B, H, D]"
            raise RuntimeError(f"Megatron Attention Q/K/V must use {layout}")
        _require_nvidia_cuda(query, "Attention")

        parallel_state = _megatron_parallel_state()
        cp_world = int(parallel_state.get_context_parallel_world_size())
        cp_rank = int(parallel_state.get_context_parallel_rank())
        tp_world = int(parallel_state.get_tensor_model_parallel_world_size())
        tp_rank = int(parallel_state.get_tensor_model_parallel_rank())
        cp_group = parallel_state.get_context_parallel_group() if cp_world > 1 else None
        operator = self._handle.get(
            query,
            topology={
                "world_size": tp_world * cp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": cp_world,
            },
        )
        operator.bind_cuda_runtime(process_group=cp_group)
        scale = float(getattr(module, "softmax_scale", query.size(-1) ** -0.5))

        def execute_sequence(
            q_ready: torch.Tensor,
            k_ready: torch.Tensor,
            v_ready: torch.Tensor,
            *,
            global_sequence_length: int,
        ) -> Any:
            positions, block_indices, block_starts, block_offsets = _megatron_zigzag_layout(
                q_ready.size(2),
                cp_rank=cp_rank,
                cp_world_size=cp_world,
            )
            position_ids = torch.tensor(
                positions,
                dtype=torch.int64,
                device=q_ready.device,
            ).repeat(q_ready.size(0), 1)
            return operator(
                q_ready,
                k_ready,
                v_ready,
                contract=_dense_attention_contract(
                    q_ready,
                    k_ready,
                    role=AttentionRole.TRAIN,
                    causal=True,
                    tp_rank=tp_rank,
                    tp_world_size=tp_world,
                    cp_rank=cp_rank,
                    cp_world_size=cp_world,
                    global_sequence_length=global_sequence_length,
                    global_block_indices=block_indices,
                    global_block_token_starts=block_starts,
                    local_block_offsets=block_offsets,
                ),
                config=AttentionAblationConfig(
                    strict_core_id=STRICT_ATTENTION_PRODUCTION_CORE_ID,
                    strict_schedule=STRICT_ATTENTION_FA4_SCHEDULE_ID,
                ),
                return_lse=True,
                communication_backend="cuda_ag_rs" if cp_world > 1 else "none",
                query_position_ids=position_ids,
                key_position_ids=position_ids,
                scale=scale,
            )

        if packed_seq_params is None:
            q_ready = query.permute(1, 2, 0, 3).contiguous()
            k_ready = key.permute(1, 2, 0, 3).contiguous()
            v_ready = value.permute(1, 2, 0, 3).contiguous()
            result = execute_sequence(
                q_ready,
                k_ready,
                v_ready,
                global_sequence_length=q_ready.size(2) * cp_world,
            )
            context = result.out.permute(2, 0, 1, 3).contiguous()
            output = context.flatten(start_dim=2)
            execution_provenance: dict[str, Any] = {
                "packed_sequence_count": 0,
                "operator": _compact_attention_provenance(result.provenance),
            }
        else:
            local_offsets, global_lengths = self._packed_layout(
                packed_seq_params,
                cp_world_size=cp_world,
                local_query_tokens=query.size(0),
                local_kv_tokens=key.size(0),
            )
            grouped_sequences: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
            for sequence_index, (start, end, global_length) in enumerate(
                zip(
                    local_offsets[:-1],
                    local_offsets[1:],
                    global_lengths,
                    strict=True,
                )
            ):
                grouped_sequences.setdefault((end - start, global_length), []).append(
                    (sequence_index, start, end)
                )

            outputs: list[torch.Tensor | None] = [None] * len(global_lengths)
            sequence_provenance: list[dict[str, Any] | None] = [None] * len(global_lengths)
            launch_group_count = 0
            for (local_length, global_length), sequences in grouped_sequences.items():
                q_ready = (
                    torch.stack([query[start:end] for _index, start, end in sequences], dim=0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                k_ready = (
                    torch.stack([key[start:end] for _index, start, end in sequences], dim=0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                v_ready = (
                    torch.stack([value[start:end] for _index, start, end in sequences], dim=0)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                result = execute_sequence(
                    q_ready,
                    k_ready,
                    v_ready,
                    global_sequence_length=global_length,
                )
                group_output = result.out.permute(0, 2, 1, 3).contiguous().flatten(start_dim=2)
                operator_provenance = _compact_attention_provenance(result.provenance)
                for batch_index, (sequence_index, _start, _end) in enumerate(sequences):
                    outputs[sequence_index] = group_output[batch_index]
                    sequence_provenance[sequence_index] = {
                        "sequence_index": sequence_index,
                        "local_tokens": local_length,
                        "global_tokens": global_length,
                        "operator": operator_provenance,
                    }
                launch_group_count += 1
            if any(item is None for item in outputs) or any(
                item is None for item in sequence_provenance
            ):
                raise RuntimeError("packed Attention failed to materialize every sequence")
            output = torch.cat(cast(list[torch.Tensor], outputs), dim=0)
            execution_provenance = {
                "packed_sequence_count": len(global_lengths),
                "launch_group_count": launch_group_count,
                "sequence_batching": "equal_length_rows",
                "sequences": cast(list[dict[str, Any]], sequence_provenance),
            }
        self._last_provenance = {
            "framework_layout": (
                "megatron_thd_packed_zigzag_cp"
                if packed_seq_params is not None
                else "megatron_sbh_zigzag_cp"
            ),
            "materialization": "owner_local_zigzag_cuda_ag_rs",
            "cp_world_size": cp_world,
            "tp_world_size": tp_world,
            "runtime_platform": "cuda",
            "triton_used": False,
            **execution_provenance,
        }
        return output


class MegatronFFNOperator:
    backend_id = FFN_BACKEND_ID

    def __init__(self, handle: SemanticOperatorHandle | None = None) -> None:
        self._handle = handle or SemanticOperatorHandle(
            target="training", semantic_op="ffn", backend_id=self.backend_id
        )
        self._last_provenance: dict[str, Any] = {}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "interface": "megatron.mlp.forward",
            "operator": self.backend_id,
            "fallback": False,
            "semantic_instance": self._handle.provenance,
            "execution": dict(self._last_provenance),
        }

    def __call__(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        per_token_scale: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        # Megatron's dense MLP path forwards padding_mask=None even when
        # no token padding is active. It is a routing-only argument and must
        # not be treated as expert-token scaling in the strict dense wrapper.
        if kwargs:
            unexpected = {
                name: value
                for name, value in kwargs.items()
                if name != "padding_mask" or value is not None
            }
            if unexpected:
                raise RuntimeError("strict dense Qwen3 FFN does not accept expert token scaling")
        if per_token_scale is not None:
            raise RuntimeError("strict dense Qwen3 FFN does not accept expert token scaling")
        _require_nvidia_cuda(hidden_states, "FFN")
        config = module.config
        if bool(getattr(config, "add_bias_linear", False)):
            raise RuntimeError("strict Qwen3 FFN requires bias-free projections")
        if not bool(getattr(config, "gated_linear_unit", False)):
            raise RuntimeError("strict Qwen3 FFN requires a gated linear unit")
        hidden_states = _fused_rms_norm_input(
            module.linear_fc1,
            hidden_states,
            "linear_fc1",
        )
        fused_gate_up = _weight(module.linear_fc1, "linear_fc1").contiguous()
        gate, up = _split_gate_up(fused_gate_up, "linear_fc1")
        down = _weight(module.linear_fc2, "linear_fc2").contiguous()
        parallel_state = _megatron_parallel_state()
        cp_world = int(parallel_state.get_context_parallel_world_size())
        tp_world = int(parallel_state.get_tensor_model_parallel_world_size())
        cp_group = parallel_state.get_context_parallel_group() if cp_world > 1 else None
        operator = self._handle.get(
            hidden_states,
            topology={
                "world_size": tp_world * cp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": cp_world,
            },
        )
        output = operator(
            hidden_states,
            gate,
            up,
            down,
            tp_group=getattr(module, "tp_group", None),
            cp_group=cp_group,
            sequence_parallel=bool(getattr(config, "sequence_parallel", False)),
            deterministic=True,
        )
        self._last_provenance = {
            "framework_layout": "megatron_sequence_parallel",
            "cp_world_size": cp_world,
            "tp_world_size": tp_world,
            "runtime_platform": "cuda",
            "actual_backend": "rlkernel.cuda.det_gemm_swiglu",
            "gemm_backend": det_gemm_backend_id(),
            "fallback": False,
            "gate_up_projection": "separate_strict_launches",
            "triton_used": False,
        }
        return output, None


def _vllm_tp_coordinates() -> tuple[int, int, Any]:
    try:
        from vllm.distributed.parallel_state import get_tp_group

        coordinator = get_tp_group()
        group = getattr(coordinator, "device_group", coordinator)
        rank = int(getattr(coordinator, "rank_in_group", 0))
        world = int(getattr(coordinator, "world_size", 1))
        return world, rank, group
    except (ImportError, AssertionError):
        return 1, 0, None


def _vllm_kv_cache_views(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return vLLM's paged cache as [blocks, block, kv_heads, head]."""

    if kv_cache.ndim == 4 and kv_cache.size(-1) == 2 * head_size:
        return kv_cache.transpose(1, 2).split(head_size, dim=-1)
    if kv_cache.ndim == 5 and kv_cache.size(0) == 2:
        key_cache, value_cache = kv_cache.unbind(0)
        return key_cache, value_cache
    raise RuntimeError(
        "vLLM FlashAttention KV cache must use " "[blocks, kv_heads, block, 2 * head_size]"
    )


class VllmAttentionOperator:
    """Materialize logical rows from vLLM paged KV, then call the wrapper."""

    backend_id = ATTENTION_BACKEND_ID

    def __init__(self, handle: SemanticOperatorHandle | None = None) -> None:
        self._handle = handle or SemanticOperatorHandle(
            target="rollout", semantic_op="attention", backend_id=self.backend_id
        )
        self._last_provenance: dict[str, Any] = {}
        self._tp_coordinates: tuple[int, int, Any] | None = None
        self._metadata_cache_key: tuple[Any, ...] | None = None
        self._metadata_cache_owners: set[int] = set()
        self._metadata_cache_value: tuple[list[dict[str, Any]], dict[str, Any]] | None = None

    def bind_inference(self) -> None:
        """Resolve the backend after vLLM has selected the worker CUDA device."""

        if self._tp_coordinates is not None:
            return
        tp_world, tp_rank, tp_group = _vllm_tp_coordinates()
        device = torch.device("cuda", torch.cuda.current_device())
        self._handle.get(
            torch.empty((1,), device=device, dtype=torch.bfloat16),
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        self._tp_coordinates = (tp_world, tp_rank, tp_group)

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "interface": "vllm.attention.forward",
            "operator": self.backend_id,
            "fallback": False,
            "semantic_instance": self._handle.provenance,
            "execution": dict(self._last_provenance),
        }

    @staticmethod
    def _metadata_tensor(metadata: Any, *names: str) -> torch.Tensor:
        for name in names:
            value = getattr(metadata, name, None)
            if isinstance(value, torch.Tensor):
                return value
        raise RuntimeError(f"vLLM Attention metadata is missing {'/'.join(names)}")

    def _materialization_groups(
        self,
        attn_metadata: Any,
        *,
        query: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        num_actual: int,
        cache_owner: Any | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build one paged launch from vLLM's graph-replayable GPU metadata."""

        query_starts_source = self._metadata_tensor(
            attn_metadata, "query_start_loc", "query_start_loc_cpu"
        )
        seq_lens_source = self._metadata_tensor(attn_metadata, "seq_lens", "seq_lens_cpu")
        if num_actual == 0:
            return [], {
                "row_count": 0,
                "query_position_range": [None, None],
                "kv_token_range": [None, None],
                "launch_group_count": 0,
                "metadata_source": "vllm_gpu",
            }

        query_starts = query_starts_source.to(device=query.device, dtype=torch.int32)
        seq_lens = seq_lens_source.to(device=query.device, dtype=torch.int32)
        query_indices = torch.arange(num_actual, dtype=torch.int32, device=query.device)
        query_ends = query_starts[1:]
        request_indices = torch.searchsorted(query_ends, query_indices, right=True).to(
            dtype=torch.long
        )
        active_queries = query_indices < query_starts[-1]
        request_indices = request_indices.clamp_max(seq_lens.numel() - 1)
        request_query_ends = query_ends.index_select(0, request_indices)
        request_seq_lens = seq_lens.index_select(0, request_indices)
        seqused_k = request_seq_lens - (request_query_ends - query_indices) + 1
        seqused_k = torch.where(
            active_queries,
            seqused_k,
            torch.ones_like(seqused_k),
        )

        max_seq_len = int(getattr(attn_metadata, "max_seq_len", block_table.size(1) * block_size))
        page_count = min(block_table.size(1), (max_seq_len + block_size - 1) // block_size)
        cache_key = (
            _tensor_cache_token(query_starts_source),
            _tensor_cache_token(seq_lens_source),
            _tensor_cache_token(block_table),
            query.device.type,
            query.device.index,
            num_actual,
            block_size,
            page_count,
        )
        owner_id = id(cache_owner) if cache_owner is not None else None
        if (
            owner_id is not None
            and self._metadata_cache_key == cache_key
            and owner_id not in self._metadata_cache_owners
            and self._metadata_cache_value is not None
        ):
            self._metadata_cache_owners.add(owner_id)
            groups, cached_summary = self._metadata_cache_value
            return groups, {**cached_summary, "metadata_reused_across_layers": True}

        pages = (
            block_table.index_select(0, request_indices)[:, :page_count]
            .to(dtype=torch.int32)
            .contiguous()
        )
        groups = [
            {
                "page_count": page_count,
                "pages": pages,
                "query_indices": query_indices,
                "query_start": 0,
                "query_count": num_actual,
                "query_contiguous": True,
                "seqused_k": seqused_k,
            }
        ]
        summary = {
            "row_count": num_actual,
            "query_position_range": "device_dynamic",
            "kv_token_range": "device_dynamic",
            "launch_group_count": 1,
            "metadata_source": "vllm_gpu",
            "metadata_reused_across_layers": False,
        }
        if owner_id is not None:
            # The first repeated layer marks the next model forward (or graph
            # capture), so dynamic metadata is re-derived instead of frozen.
            self._metadata_cache_key = cache_key
            self._metadata_cache_owners = {owner_id}
            self._metadata_cache_value = (groups, summary)
        return groups, summary

    def __call__(
        self,
        impl: Any,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise RuntimeError("strict vLLM Attention does not support quantized output")
        if attn_metadata is None:
            if output is None:
                raise RuntimeError("vLLM profiling Attention requires an output buffer")
            return output.zero_()
        if query.ndim != 3:
            raise RuntimeError("vLLM query must use [tokens, heads, head_dim]")
        _require_nvidia_cuda(query, "Attention")
        key_cache, value_cache = _vllm_kv_cache_views(
            kv_cache,
            head_size=int(impl.head_size),
        )
        if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
            raise RuntimeError("strict vLLM Attention requires an unquantized KV cache")

        block_table = self._metadata_tensor(attn_metadata, "block_table", "block_table_tensor")
        num_actual = int(getattr(attn_metadata, "num_actual_tokens", query.size(0)))
        if num_actual < 0 or num_actual > query.size(0):
            raise RuntimeError("vLLM num_actual_tokens is outside the query buffer")
        if output is None:
            output = torch.empty(
                (query.size(0), impl.num_heads * impl.head_size),
                dtype=query.dtype,
                device=query.device,
            )
        if output.size(0) < num_actual:
            raise RuntimeError("vLLM output buffer is smaller than num_actual_tokens")
        output_heads = output.view(output.size(0), impl.num_heads, impl.head_size)
        block_size = key_cache.size(1)
        if self._tp_coordinates is None:
            self._tp_coordinates = _vllm_tp_coordinates()
        tp_world, tp_rank, tp_group = self._tp_coordinates
        operator = self._handle.get(
            query,
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        runtime = operator.bind_cuda_runtime()
        groups, metadata_summary = self._materialization_groups(
            attn_metadata,
            query=query,
            block_table=block_table,
            block_size=block_size,
            num_actual=num_actual,
            cache_owner=layer,
        )
        last_operator_provenance: dict[str, Any] = {}
        next_query_row = 0
        for group in groups:
            if not group["query_contiguous"] or group["query_start"] != next_query_row:
                break
            next_query_row += int(group["query_count"])
        direct_output_buffer = bool(groups) and next_query_row == num_actual
        if not direct_output_buffer:
            output.zero_()
        elif num_actual < output.size(0):
            output[num_actual:].zero_()
        for group in groups:
            page_count = int(group["page_count"])
            pages = group["pages"]
            query_indices = group["query_indices"]
            if group["query_contiguous"]:
                query_start = int(group["query_start"])
                query_count = int(group["query_count"])
                q_ready = query.narrow(0, query_start, query_count).unsqueeze(2)
            else:
                q_ready = query.index_select(0, query_indices).unsqueeze(2).contiguous()
            result = runtime.forward_paged_with_lse(
                q_ready,
                key_cache,
                value_cache,
                page_table=pages,
                seqused_k=group["seqused_k"],
                max_seqlen_k=page_count * block_size,
                scale=float(impl.scale),
            )
            result_output = result.out.squeeze(2)
            if group["query_contiguous"]:
                output_heads.narrow(0, int(group["query_start"]), int(group["query_count"])).copy_(
                    result_output
                )
            else:
                output_heads.index_copy_(0, query_indices, result.out.squeeze(2))
            last_operator_provenance = _compact_attention_provenance(result.provenance)
        self._last_provenance = {
            "framework_layout": "vllm_paged_kv",
            "materialization": "direct_paged_fa4",
            "tp_world_size": tp_world,
            "tp_group_bound": tp_group is not None,
            "runtime_platform": "cuda",
            "triton_used": False,
            "direct_output_buffer": direct_output_buffer,
            **metadata_summary,
            "operator": last_operator_provenance,
        }
        return output


class VllmFFNOperator:
    backend_id = FFN_BACKEND_ID

    def __init__(self, handle: SemanticOperatorHandle | None = None) -> None:
        self._handle = handle or SemanticOperatorHandle(
            target="rollout", semantic_op="ffn", backend_id=self.backend_id
        )
        self._last_provenance: dict[str, Any] = {}
        self._packed_inference_binding: tuple[int, int, str] | None = None
        self._tp_coordinates: tuple[int, int, Any] | None = None

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "interface": "vllm.qwen3.mlp.forward",
            "operator": self.backend_id,
            "fallback": False,
            "semantic_instance": self._handle.provenance,
            "execution": dict(self._last_provenance),
        }

    def bind_packed_inference(self, module: Any) -> tuple[int, int]:
        """Bind vLLM's TP group before torch.compile captures the model."""

        fused_gate_up = _weight(module.gate_up_proj, "gate_up_proj")
        down = _weight(module.down_proj, "down_proj")
        tp_world, tp_rank, tp_group = _vllm_tp_coordinates()
        self._tp_coordinates = (tp_world, tp_rank, tp_group)
        operator = self._handle.get(
            fused_gate_up,
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        prepare = getattr(operator, "prepare_packed_inference", None)
        if not callable(prepare):
            raise RuntimeError("strict FFN backend lacks packed inference preparation")
        collective_handle, bound_tp_world = prepare(
            fused_gate_up,
            down,
            tp_group=tp_group,
        )
        backend_id = "deterministic_all_reduce.ipc_localized_fixed_tree.v1"
        resolve_backend = getattr(operator, "packed_inference_backend_id", None)
        if callable(resolve_backend) and collective_handle:
            backend_id = str(resolve_backend(collective_handle))
        self._packed_inference_binding = (
            collective_handle,
            bound_tp_world,
            backend_id,
        )
        self._set_runtime_provenance(tp_world, backend_id)
        return collective_handle, bound_tp_world

    def _set_runtime_provenance(
        self, tp_world: int, deterministic_all_reduce_backend: str = "unbound"
    ) -> None:
        self._last_provenance = {
            "framework_layout": "vllm_tensor_parallel",
            "tp_world_size": tp_world,
            "runtime_platform": "cuda",
            "actual_backend": "rlkernel.cuda.det_gemm_swiglu",
            "gemm_backend": det_gemm_backend_id(),
            "fallback": False,
            "gate_up_projection": "packed_single_launch",
            "deterministic_all_reduce_backend": deterministic_all_reduce_backend,
            "triton_used": False,
        }

    def __call__(self, module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_nvidia_cuda(hidden_states, "FFN")
        fused_gate_up = _weight(module.gate_up_proj, "gate_up_proj").contiguous()
        down = _weight(module.down_proj, "down_proj").contiguous()
        if self._tp_coordinates is None:
            self._tp_coordinates = _vllm_tp_coordinates()
        tp_world, _tp_rank, tp_group = self._tp_coordinates
        operator = self._handle.get(
            hidden_states,
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        bound_backend = "unbound"
        if self._packed_inference_binding is not None:
            bound_backend = self._packed_inference_binding[2]
        self._set_runtime_provenance(tp_world, bound_backend)
        if self._packed_inference_binding is not None:
            collective_handle, bound_tp_world, _ = self._packed_inference_binding
            if bound_tp_world != tp_world:
                raise RuntimeError("packed rollout FFN TP topology changed after binding")
            output = operator.packed_inference(
                hidden_states,
                fused_gate_up,
                down,
                collective_handle=collective_handle,
                tp_world_size=bound_tp_world,
            )
        else:
            if torch._dynamo.is_compiling():
                raise RuntimeError("packed rollout FFN was not bound before torch.compile capture")
            gate, up = _split_gate_up(fused_gate_up, "gate_up_proj")
            output = operator(
                hidden_states,
                gate,
                up,
                down,
                tp_group=tp_group,
                sequence_parallel=False,
                deterministic=True,
            )
        return output


class MegatronLogpOperator:
    """Require CUDA while reusing the structural Vime Logp provider."""

    def __init__(
        self,
        provider: Any,
        *,
        linear_logp: LinearLogpWrapper | None = None,
    ) -> None:
        self._provider = provider
        self._linear_logp = linear_logp
        self._last_provenance: dict[str, Any] = {}

    @property
    def backend_id(self) -> str:
        if self._linear_logp is not None:
            return self._linear_logp.backend_id
        return LOGP_BACKEND_ID

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self._last_provenance)

    def __call__(self, request: Any) -> Any:
        logits = getattr(request, "logits", None)
        context = getattr(request, "context", None)
        hidden = getattr(request, "hidden", None)
        if not isinstance(hidden, torch.Tensor):
            hidden = getattr(context, "hidden", None)
        strict = strict_contract_enabled()
        if isinstance(hidden, torch.Tensor):
            if self._linear_logp is None:
                raise RuntimeError("Megatron linear_logp route is not installed")
            _require_nvidia_cuda(hidden, "linear_logp")
            result = self._provider(request, linear_logp=self._linear_logp)
            self._last_provenance = {
                "interface": "vime.selected_logprob_provider",
                "operator": self.backend_id,
                "actual_backend": self.backend_id,
                "fallback": False,
                "runtime_platform": "cuda",
                "triton_used": False,
                "provider": dict(getattr(result, "provenance", {})),
                "linear_logp": dict(self._linear_logp.provenance),
                "logits_materialized": False,
            }
            return result
        if strict:
            raise RuntimeError(
                "strict Megatron linear_logp request is missing hidden/LM-head structural inputs"
            )
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Megatron Logp request must expose logits or hidden")
        _require_nvidia_cuda(logits, "Logp")
        result = self._provider(request)
        self._last_provenance = {
            "interface": "vime.selected_logprob_provider",
            "operator": self.backend_id,
            "actual_backend": self.backend_id,
            "fallback": False,
            "runtime_platform": "cuda",
            "triton_used": False,
            "provider": dict(getattr(result, "provenance", {})),
        }
        return result


def _full_vocab_contract(logits: torch.Tensor) -> LogprobContract:
    dtype = {
        torch.bfloat16: LogprobDType.BF16,
        torch.float16: LogprobDType.FP16,
        torch.float32: LogprobDType.FP32,
    }.get(logits.dtype)
    if dtype is None:
        raise RuntimeError(f"unsupported vLLM logit dtype {logits.dtype}")
    tokens, vocab = logits.shape
    return LogprobContract(
        role=LogprobRole.INFER,
        dtype=dtype,
        mask=MaskSpec(num_tokens=tokens, active_mask=(True,) * tokens),
        sharding=LogprobShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, vocab),),
            real_vocab_size=vocab,
            padded_vocab_size=vocab,
        ),
        reduction=LogprobReductionSpec(),
    )


def _diagnostic_vocab_contract(
    logits: torch.Tensor,
    *,
    real_vocab_size: int | None,
    padded_vocab_size: int | None,
) -> LogprobContract | None:
    if real_vocab_size is None or padded_vocab_size is None:
        return None
    if real_vocab_size <= 0 or padded_vocab_size <= 0:
        return None
    if real_vocab_size > padded_vocab_size:
        return None
    tokens, vocab = logits.shape
    if padded_vocab_size != vocab:
        return None
    if padded_vocab_size % DEFAULT_NUM_VOCAB_TILES:
        return None
    dtype = {
        torch.bfloat16: LogprobDType.BF16,
        torch.float16: LogprobDType.FP16,
        torch.float32: LogprobDType.FP32,
    }.get(logits.dtype)
    if dtype is None:
        return None
    return LogprobContract(
        role=LogprobRole.INFER,
        dtype=dtype,
        mask=MaskSpec(num_tokens=tokens, active_mask=(True,) * tokens),
        sharding=LogprobShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, padded_vocab_size),),
            real_vocab_size=real_vocab_size,
            padded_vocab_size=padded_vocab_size,
        ),
        reduction=LogprobReductionSpec(),
    )


class VllmLogpOperator:
    """Keep sampling, then replace sampled-token logp with the selected contract."""

    def __init__(
        self,
        native_forward: Any,
        *,
        worker_sampler: bool = False,
        strict_linear_logp: bool = False,
    ) -> None:
        self._native_forward = native_forward
        self._worker_sampler = worker_sampler
        self._strict_linear_logp = strict_linear_logp
        self._linear_logp = LinearLogpWrapper() if strict_linear_logp else None
        self._last_provenance: dict[str, Any] = {}

    @property
    def backend_id(self) -> str:
        if self._strict_linear_logp:
            assert self._linear_logp is not None
            return self._linear_logp.backend_id
        return LOGP_BACKEND_ID

    @property
    def provenance(self) -> Mapping[str, Any]:
        return dict(self._last_provenance)

    def _replace_sampled_value(
        self,
        result: Any,
        *,
        token_ids: torch.Tensor,
        selected: torch.Tensor,
        provenance: Mapping[str, Any],
    ) -> Any:
        logprobs_tensors = getattr(result, "logprobs_tensors", None)
        if logprobs_tensors is None:
            raise RuntimeError(
                "strict vLLM Logp requires sampled-token logprob tensors; "
                "native sampling returned none"
            )
        ids = logprobs_tensors.logprob_token_ids
        values = logprobs_tensors.logprobs.clone()
        if selected.numel() != token_ids.numel():
            raise RuntimeError(
                "strict vLLM logp selected output is not aligned with sampled tokens"
            )
        matches = ids == token_ids.unsqueeze(1)
        every_sample_present = matches.any(dim=1).all()
        if every_sample_present.is_cuda:
            torch._assert_async(
                every_sample_present,
                "vLLM logprob result does not contain every sampled token",
            )
        elif not bool(every_sample_present):
            raise RuntimeError("vLLM logprob result does not contain every sampled token")
        # vLLM prepends the sampled token and then appends top-k tokens. When
        # the sample is also in top-k, its API conversion keeps the last
        # duplicate, so every matching column must carry the strict value.
        values = torch.where(matches, selected.unsqueeze(1), values)
        self._last_provenance = {
            **dict(provenance),
            "interface": "vllm.sampler.selected_logprob",
            "operator": self.backend_id,
            "actual_backend": self.backend_id,
            "fallback": False,
            "sampled_token_ids": _tensor_metadata(token_ids),
            "logprobs_shape": list(values.shape),
            "logprob_token_ids_shape": list(ids.shape),
            "strict_selected_logp": _tensor_metadata(selected),
            "native_reference_compared": False,
        }
        if hasattr(logprobs_tensors, "_replace"):
            updated_tensors = logprobs_tensors._replace(logprobs=values)
        else:
            updated_tensors = replace(logprobs_tensors, logprobs=values)
        return replace(result, logprobs_tensors=updated_tensors)

    def __call__(
        self,
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ) -> Any:
        # Native sampling owns logits; strict selected-logp preserves its raw
        # local shard before the sampler applies in-place transformations.
        source_logits = logits
        _require_nvidia_cuda(source_logits, "Logp")
        context = None
        local_logits = None
        if self._strict_linear_logp:
            context = take_rollout_linear_logp_context()
            if source_logits.ndim != 2:
                raise RuntimeError("vLLM sampler logits must be [tokens, vocab]")
            if context.hidden.size(0) != source_logits.size(0):
                raise RuntimeError(
                    "strict rollout linear_logp hidden/logits row mismatch: "
                    f"{context.hidden.size(0)} != {source_logits.size(0)}"
                )
            if source_logits.size(1) < context.real_vocab_size:
                raise RuntimeError(
                    "strict rollout source logits do not cover the real vocabulary: "
                    f"{source_logits.size(1)} < {context.real_vocab_size}"
                )
            local_vocab = int(context.lm_head_weight.size(0))
            available = max(
                0,
                min(
                    local_vocab,
                    source_logits.size(1) - context.vocab_start_index,
                ),
            )
            # Preserve raw model logits before vLLM's sampler transforms its
            # input in place (temperature, penalties, and masking).
            if available == local_vocab:
                # Rank 0 normally has a complete local shard. Narrowing first
                # avoids a fill kernel followed by a second device copy.
                local_logits = source_logits.narrow(
                    1,
                    context.vocab_start_index,
                    local_vocab,
                ).contiguous()
            else:
                # The final TP shard may include padded rows absent from the
                # serving logits; retain the exact -inf padding contract.
                local_logits = source_logits.new_full(
                    (source_logits.size(0), local_vocab),
                    float("-inf"),
                )
                if available:
                    local_logits[:, :available].copy_(
                        source_logits.narrow(
                            1,
                            context.vocab_start_index,
                            available,
                        )
                    )
        if self._worker_sampler:
            result = self._native_forward(sampler, logits, sampling_metadata)
        else:
            result = self._native_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token=predict_bonus_token,
                logprobs_mode_override=logprobs_mode_override,
            )
        logprobs_tensors = getattr(result, "logprobs_tensors", None)
        if logprobs_tensors is None:
            raise RuntimeError(
                "strict vLLM Logp requires sampled-token logprob tensors; "
                "native sampling returned none"
            )
        token_ids = result.sampled_token_ids.reshape(-1).to(torch.long)

        if self._strict_linear_logp:
            assert context is not None and local_logits is not None
            if context.hidden.size(0) != token_ids.numel():
                raise RuntimeError(
                    "strict rollout linear_logp hidden/sample row mismatch: "
                    f"{context.hidden.size(0)} != {token_ids.numel()}"
                )
            assert self._linear_logp is not None
            selected = self._linear_logp.from_local_logits(
                local_logits,
                token_ids,
                tp_group=context.tp_group,
                vocab_start_index=context.vocab_start_index,
                global_vocab_size=context.global_vocab_size,
                real_vocab_size=context.real_vocab_size,
                temperature=float(os.getenv("RL_KERNEL_VLLM_TEMPERATURE", "1.0")),
                target="rollout",
            )
            strict_provenance = self._linear_logp.provenance
            if (
                strict_provenance.get("deterministic_linear_logp") is not True
                or strict_provenance.get("actual_backend") != self._linear_logp.backend_id
                or strict_provenance.get("strict_entrypoint")
                != "sm90_deterministic_logp_from_local_logits_tp"
            ):
                raise RuntimeError(
                    "strict vLLM rollout linear_logp did not execute the deterministic "
                    "linear-logp entry point"
                )
            provenance = {
                **dict(strict_provenance),
                "runtime_platform": "cuda",
                "triton_used": False,
                "execution": {
                    "role": "vllm_rollout_linear_logprob",
                    "strict_backend": True,
                    "sampling_logits_source": "rlkernel_det_gemm_vllm",
                    "logits_materialized": True,
                    "padded_lm_head_alignment": True,
                    "duplicate_lm_head_gemm": False,
                },
                "source_logits_shape": list(source_logits.shape),
                "source_logits_dtype": _dtype_name(source_logits),
            }
            return self._replace_sampled_value(
                result,
                token_ids=token_ids,
                selected=selected,
                provenance=provenance,
            )

        if source_logits.ndim != 2:
            raise RuntimeError("vLLM sampler logits must be [tokens, vocab]")
        if source_logits.size(1) % DEFAULT_NUM_VOCAB_TILES:
            raise RuntimeError(
                "strict vLLM logp requires the padded vocabulary to be divisible by "
                f"{DEFAULT_NUM_VOCAB_TILES}"
            )
        contract = _full_vocab_contract(source_logits)
        from rl_engine.kernels.registry import kernel_registry

        dispatch = kernel_registry.get_logprob_op(contract, requested_backend=self.backend_id)
        if (
            dispatch.provenance["actual_backend"] != self.backend_id
            or dispatch.provenance["fallback"]
        ):
            raise RuntimeError("strict vLLM Logp dispatch changed backend")
        selected, _lse = dispatch.op(
            source_logits,
            token_ids,
            contract=contract,
            num_vocab_tiles=DEFAULT_NUM_VOCAB_TILES,
            deterministic=True,
        )
        real_vocab_env = _optional_env_int("RL_KERNEL_VLLM_REAL_VOCAB_SIZE")
        padded_vocab_env = _optional_env_int("RL_KERNEL_VLLM_PADDED_VOCAB_SIZE")
        masked_vocab_diagnostic: dict[str, Any] = {
            "env_real_vocab_size": real_vocab_env,
            "env_padded_vocab_size": padded_vocab_env,
            "status": (
                "not_requested"
                if real_vocab_env is None and padded_vocab_env is None
                else "skipped"
            ),
        }
        diagnostic_contract = _diagnostic_vocab_contract(
            source_logits,
            real_vocab_size=real_vocab_env,
            padded_vocab_size=padded_vocab_env,
        )
        if diagnostic_contract is not None:
            try:
                diagnostic_dispatch = kernel_registry.get_logprob_op(
                    diagnostic_contract, requested_backend=self.backend_id
                )
                masked_selected, _masked_lse = diagnostic_dispatch.op(
                    source_logits,
                    token_ids,
                    contract=diagnostic_contract,
                    num_vocab_tiles=DEFAULT_NUM_VOCAB_TILES,
                    deterministic=True,
                )
                masked_vocab_diagnostic.update(
                    {
                        "status": "computed_not_applied",
                        "contract_real_vocab_size": diagnostic_contract.sharding.real_vocab_size,
                        "contract_padded_vocab_size": (
                            diagnostic_contract.sharding.padded_vocab_size
                        ),
                        "selected_stats": _tensor_debug_stats(masked_selected),
                        "diff_vs_native_selected": _diff_debug_stats(
                            masked_selected,
                            logprobs_tensors.logprobs[
                                torch.arange(
                                    logprobs_tensors.logprobs.size(0),
                                    device=logprobs_tensors.logprobs.device,
                                ),
                                (logprobs_tensors.logprob_token_ids == token_ids.unsqueeze(1))
                                .to(torch.int64)
                                .argmax(dim=1),
                            ],
                        ),
                        "diff_vs_current_rlkernel_selected": _diff_debug_stats(
                            masked_selected, selected
                        ),
                    }
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                masked_vocab_diagnostic.update(
                    {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                )
        elif real_vocab_env is not None or padded_vocab_env is not None:
            masked_vocab_diagnostic.update(
                {
                    "source_logits_vocab_width": int(source_logits.size(1)),
                    "reason": (
                        "env metadata must be positive, real<=padded, and "
                        "padded_vocab_size must match vLLM logits width"
                    ),
                }
            )

        provenance = {
            **dict(dispatch.provenance),
            "runtime_platform": "cuda",
            "triton_used": False,
            "source_logits_shape": list(source_logits.shape),
            "source_logits_dtype": _dtype_name(source_logits),
            "contract_real_vocab_size": contract.sharding.real_vocab_size,
            "contract_padded_vocab_size": contract.sharding.padded_vocab_size,
            "masked_vocab_diagnostic": masked_vocab_diagnostic,
        }
        return self._replace_sampled_value(
            result,
            token_ids=token_ids,
            selected=selected,
            provenance=provenance,
        )


__all__ = [
    "MegatronAttentionOperator",
    "MegatronFFNOperator",
    "MegatronLogpOperator",
    "SemanticOperatorHandle",
    "VllmAttentionOperator",
    "VllmFFNOperator",
    "VllmLogpOperator",
]
