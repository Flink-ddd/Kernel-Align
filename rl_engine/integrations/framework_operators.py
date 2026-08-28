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
from rl_engine.integrations.linear_logp import (
    LinearLogpWrapper,
    take_rollout_linear_logp_context,
)
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
from rl_engine.kernels.attention_contract import (
    SplitKVSpec,
)
from rl_engine.kernels.logprob_contract import (
    LogprobContract,
    LogprobDType,
    LogprobRole,
    MaskSpec,
)
from rl_engine.kernels.logprob_contract import ReductionSpec as LogprobReductionSpec
from rl_engine.kernels.logprob_contract import ShardingSpec as LogprobShardingSpec
from rl_engine.kernels.ops.pytorch.attention.ablation import AttentionAblationConfig
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import BACKEND_ID as LOGP_BACKEND_ID
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import (
    DEFAULT_NUM_VOCAB_TILES,
)
from rl_engine.kernels.registry import kernel_registry
from rl_engine.kernels.semantic_registry import OperatorRequirements

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


def _tensor_debug_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    stats: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": _dtype_name(detached),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }
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
        self._requirements: OperatorRequirements | None = None
        self._provenance: dict[str, Any] | None = None
        self._lock = Lock()

    def get(
        self,
        tensor: torch.Tensor,
        *,
        topology: Mapping[str, Any],
        factory_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        requirements = OperatorRequirements(
            device=_device_name(tensor),
            dtype=_dtype_name(tensor),
            topology=topology,
            alignment_properties={"deterministic": True},
        )
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
            if self._requirements is not None and (
                requirements.device != self._requirements.device
                or requirements.dtype != self._requirements.dtype
            ):
                raise RuntimeError(
                    f"{self.semantic_op} runtime device/dtype changed after resolution"
                )
            return self._instance
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
        self._requirements = requirements
        self._provenance = provenance.to_dict()
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

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
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
            local_offsets, global_lengths = _packed_local_sequence_layout(
                packed_seq_params,
                cp_world_size=cp_world,
                local_query_tokens=query.size(0),
                local_kv_tokens=key.size(0),
            )
            outputs: list[torch.Tensor] = []
            sequence_provenance: list[dict[str, Any]] = []
            for sequence_index, (start, end, global_length) in enumerate(
                zip(
                    local_offsets[:-1],
                    local_offsets[1:],
                    global_lengths,
                    strict=True,
                )
            ):
                q_ready = query[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
                k_ready = key[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
                v_ready = value[start:end].permute(1, 0, 2).unsqueeze(0).contiguous()
                result = execute_sequence(
                    q_ready,
                    k_ready,
                    v_ready,
                    global_sequence_length=global_length,
                )
                outputs.append(
                    result.out.squeeze(0).permute(1, 0, 2).contiguous().flatten(start_dim=1)
                )
                sequence_provenance.append(
                    {
                        "sequence_index": sequence_index,
                        "local_tokens": end - start,
                        "global_tokens": global_length,
                        "operator": _compact_attention_provenance(result.provenance),
                    }
                )
            output = torch.cat(outputs, dim=0)
            execution_provenance = {
                "packed_sequence_count": len(global_lengths),
                "sequences": sequence_provenance,
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
        gate, up = _split_gate_up(_weight(module.linear_fc1, "linear_fc1"), "linear_fc1")
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
        self._prime_semantic_handle()

    def _prime_semantic_handle(self) -> None:
        # vLLM wraps model execution in torch.compile. Resolve the semantic
        # descriptor before graph capture so JSON/inspect-based provenance
        # never runs inside Dynamo's fullgraph region.
        if not torch.cuda.is_available():
            return
        try:
            tp_world, _rank, _group = _vllm_tp_coordinates()
            self._handle.get(
                torch.empty((1,), device="cuda", dtype=torch.bfloat16),
                topology={
                    "world_size": tp_world,
                    "tensor_parallel_size": tp_world,
                    "context_parallel_size": 1,
                },
            )
        except (RuntimeError, ValueError, ImportError):
            # API-server/plugin construction can precede worker CUDA setup;
            # the worker retries during initialization before graph capture.
            return

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
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
        del layer, key, value
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

        query_starts = self._metadata_tensor(
            attn_metadata, "query_start_loc", "query_start_loc_cpu"
        ).to(device="cpu", dtype=torch.long)
        seq_lens = self._metadata_tensor(attn_metadata, "seq_lens").to(
            device="cpu", dtype=torch.long
        )
        block_table = self._metadata_tensor(attn_metadata, "block_table", "block_table_tensor").to(
            device="cpu", dtype=torch.long
        )
        num_actual = int(getattr(attn_metadata, "num_actual_tokens", query.size(0)))
        if output is None:
            output = torch.empty(
                (query.size(0), impl.num_heads * impl.head_size),
                dtype=query.dtype,
                device=query.device,
            )
        output.zero_()
        output_heads = output.view(output.size(0), impl.num_heads, impl.head_size)
        block_size = key_cache.size(1)
        tp_world, tp_rank, tp_group = _vllm_tp_coordinates()
        operator = self._handle.get(
            query,
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        operator.bind_cuda_runtime()
        row_count = 0
        first_query_position: int | None = None
        last_query_position: int | None = None
        min_kv_tokens: int | None = None
        max_kv_tokens: int | None = None
        last_operator_provenance: dict[str, Any] = {}
        for request_index in range(seq_lens.numel()):
            q_start = int(query_starts[request_index])
            q_end = min(int(query_starts[request_index + 1]), num_actual)
            if q_start >= q_end:
                continue
            final_seq_len = int(seq_lens[request_index])
            first_query_position = final_seq_len - (q_end - q_start)
            for token_offset, query_index in enumerate(range(q_start, q_end)):
                kv_len = first_query_position + token_offset + 1
                page_count = (kv_len + block_size - 1) // block_size
                page_ids = block_table[request_index, :page_count].tolist()
                if any(page < 0 for page in page_ids):
                    raise RuntimeError("vLLM block table contains an unallocated page")
                pages = torch.tensor(page_ids, device=key_cache.device, dtype=torch.long)
                k_row = key_cache.index_select(0, pages).reshape(
                    -1, impl.num_kv_heads, impl.head_size
                )[:kv_len]
                v_row = value_cache.index_select(0, pages).reshape(
                    -1, impl.num_kv_heads, impl.head_size
                )[:kv_len]
                q_ready = query[query_index : query_index + 1].permute(1, 0, 2).unsqueeze(0)
                k_ready = k_row.permute(1, 0, 2).unsqueeze(0).contiguous()
                v_ready = v_row.permute(1, 0, 2).unsqueeze(0).contiguous()
                query_positions = torch.tensor(
                    [[kv_len - 1]],
                    dtype=torch.int64,
                    device=query.device,
                )
                key_positions = torch.arange(
                    kv_len,
                    dtype=torch.int64,
                    device=query.device,
                ).unsqueeze(0)
                result = operator(
                    q_ready.contiguous(),
                    k_ready,
                    v_ready,
                    contract=_dense_attention_contract(
                        q_ready,
                        k_ready,
                        role=AttentionRole.INFER,
                        # The causal prefix is already materialized as one dense row.
                        causal=False,
                        tp_rank=tp_rank,
                        tp_world_size=tp_world,
                        mode=AttentionMode.CHUNKED_PREFILL,
                        global_sequence_length=kv_len,
                        global_block_token_starts=(kv_len - 1,),
                    ),
                    config=AttentionAblationConfig(
                        strict_core_id=STRICT_ATTENTION_PRODUCTION_CORE_ID,
                        strict_schedule=STRICT_ATTENTION_FA4_SCHEDULE_ID,
                    ),
                    return_lse=True,
                    query_position_ids=query_positions,
                    key_position_ids=key_positions,
                    scale=float(impl.scale),
                )
                output_heads[query_index].copy_(result.out[0, :, 0, :])
                query_position = kv_len - 1
                row_count += 1
                first_query_position = (
                    query_position
                    if first_query_position is None
                    else min(first_query_position, query_position)
                )
                last_query_position = (
                    query_position
                    if last_query_position is None
                    else max(last_query_position, query_position)
                )
                min_kv_tokens = kv_len if min_kv_tokens is None else min(min_kv_tokens, kv_len)
                max_kv_tokens = kv_len if max_kv_tokens is None else max(max_kv_tokens, kv_len)
                last_operator_provenance = _compact_attention_provenance(result.provenance)
        self._last_provenance = {
            "framework_layout": "vllm_paged_kv",
            "materialization": "one_causal_prefix_per_query_row",
            "tp_world_size": tp_world,
            "tp_group_bound": tp_group is not None,
            "runtime_platform": "cuda",
            "triton_used": False,
            "row_count": row_count,
            "query_position_range": [first_query_position, last_query_position],
            "kv_token_range": [min_kv_tokens, max_kv_tokens],
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
        self._prime_semantic_handle()

    def _prime_semantic_handle(self) -> None:
        # vLLM wraps model execution in torch.compile. Resolve the semantic
        # descriptor before graph capture so JSON/inspect-based provenance
        # never runs inside Dynamo's fullgraph region.
        if not torch.cuda.is_available():
            return
        try:
            tp_world, _rank, _group = _vllm_tp_coordinates()
            self._handle.get(
                torch.empty((1,), device="cuda", dtype=torch.bfloat16),
                topology={
                    "world_size": tp_world,
                    "tensor_parallel_size": tp_world,
                    "context_parallel_size": 1,
                },
            )
        except (RuntimeError, ValueError, ImportError):
            # API-server/plugin construction can precede worker CUDA setup;
            # the worker retries during initialization before graph capture.
            return

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "semantic_instance": self._handle.provenance,
            "execution": dict(self._last_provenance),
        }

    def __call__(self, module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_nvidia_cuda(hidden_states, "FFN")
        gate, up = _split_gate_up(_weight(module.gate_up_proj, "gate_up_proj"), "gate_up_proj")
        down = _weight(module.down_proj, "down_proj").contiguous()
        tp_world, _tp_rank, tp_group = _vllm_tp_coordinates()
        operator = self._handle.get(
            hidden_states,
            topology={
                "world_size": tp_world,
                "tensor_parallel_size": tp_world,
                "context_parallel_size": 1,
            },
        )
        output = operator(
            hidden_states,
            gate,
            up,
            down,
            tp_group=tp_group,
            sequence_parallel=False,
            deterministic=True,
        )
        self._last_provenance = {
            "framework_layout": "vllm_tensor_parallel",
            "tp_world_size": tp_world,
            "runtime_platform": "cuda",
            "actual_backend": "rlkernel.cuda.det_gemm_swiglu",
            "triton_used": False,
        }
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
        hidden = getattr(context, "hidden", None)
        strict = os.getenv("VIME_RL_KERNEL_STRICT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if isinstance(hidden, torch.Tensor):
            if self._linear_logp is None:
                raise RuntimeError("Megatron linear_logp route is not installed")
            _require_nvidia_cuda(hidden, "linear_logp")
            result = self._provider(request, linear_logp=self._linear_logp)
            self._last_provenance = {
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
        if not bool(matches.any(dim=1).all().item()):
            raise RuntimeError("vLLM logprob result does not contain every sampled token")
        columns = matches.to(torch.int64).argmax(dim=1)
        native_selected = values[
            torch.arange(values.size(0), device=values.device), columns
        ].clone()
        # vLLM prepends the sampled token and then appends top-k tokens. When
        # the sample is also in top-k, its API conversion keeps the last
        # duplicate, so every matching column must carry the strict value.
        values = torch.where(matches, selected.unsqueeze(1), values)
        self._last_provenance = {
            **dict(provenance),
            "sampled_token_ids": _tensor_debug_stats(token_ids),
            "logprobs_shape": list(values.shape),
            "logprob_token_ids_shape": list(ids.shape),
            "native_selected_logp_stats": _tensor_debug_stats(native_selected),
            "rlkernel_selected_logp_stats": _tensor_debug_stats(selected),
            "native_vs_rlkernel_selected_diff": _diff_debug_stats(native_selected, selected),
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
        source_logits = logits.clone()
        _require_nvidia_cuda(source_logits, "Logp")
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
            context = take_rollout_linear_logp_context()
            if source_logits.ndim != 2:
                raise RuntimeError("vLLM sampler logits must be [tokens, vocab]")
            if context.hidden.size(0) != source_logits.size(0):
                raise RuntimeError(
                    "strict rollout linear_logp hidden/logits row mismatch: "
                    f"{context.hidden.size(0)} != {source_logits.size(0)}"
                )
            if context.hidden.size(0) != token_ids.numel():
                raise RuntimeError(
                    "strict rollout linear_logp hidden/sample row mismatch: "
                    f"{context.hidden.size(0)} != {token_ids.numel()}"
                )
            assert self._linear_logp is not None
            selected = self._linear_logp(
                context.hidden,
                context.lm_head_weight,
                token_ids,
                context.lm_head_bias,
                tp_group=context.tp_group,
                vocab_start_index=context.vocab_start_index,
                global_vocab_size=context.global_vocab_size,
                real_vocab_size=context.real_vocab_size,
                temperature=float(os.getenv("RL_KERNEL_VLLM_TEMPERATURE", "1.0")),
                target="rollout",
            )
            provenance = {
                **dict(self._linear_logp.provenance),
                "runtime_platform": "cuda",
                "triton_used": False,
                "execution": {
                    "role": "vllm_rollout_linear_logprob",
                    "strict_backend": True,
                    "sampling_logits_source": "native_vllm",
                    "logits_materialized": True,
                    "padded_lm_head_alignment": True,
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
