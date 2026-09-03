# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic collectives for CUDA IPC and ROCm rank-ordered transport.

ROCm uses HIP IPC where it wins and RCCL otherwise. Reduction arithmetic stays
outside RCCL and follows the same fixed balanced rank tree on every rank.
"""

from __future__ import annotations

import socket
import threading
from types import TracebackType
from typing import Any

import torch
import torch.distributed as dist

_SUPPORTED_WORLD_SIZES = (1, 2, 4, 8)
_DEFAULT_MAX_SIZE_BYTES = 64 * 1024 * 1024
# Packing two independent lanes saves a collective launch for small tensors,
# but doubles the message size seen by RCCL. On MI300X, separate AllGather
# transports win once the packed payload reaches the multi-megabyte regime.
# Keep the crossover explicit and easy to retune with new RCCL releases.
_PACKED_REDUCE_SCATTER_MAX_BYTES = 8 * 1024 * 1024
_ROCM_IPC_DIRECT_ALL_REDUCE_MAX_BYTES = 768 * 1024
_ROCM_IPC_SHARDED_ALL_REDUCE_MIN_BYTES = 2176 * 1024
_ROCM_IPC_ALL_GATHER_MAX_BYTES = 256 * 1024
_COLLECTIVE_STAGING_FRAMES = 3
_COLLECTIVE_FRAME_METADATA_BYTES = 3 * 8
_REDUCTION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_COLLECTIVES: dict[tuple[int, int, int, int], Any] = {}
DETERMINISTIC_ALL_REDUCE_OP = "rl_kernel::deterministic_all_reduce_"


@torch.library.custom_op(DETERMINISTIC_ALL_REDUCE_OP, mutates_args={"input"})
def _deterministic_all_reduce_(input: torch.Tensor, collective_handle: int) -> None:
    """Expose the stateful IPC reduction as an explicit graph mutation."""

    from rl_engine import _C

    _C.deterministic_collective_all_reduce_fused(collective_handle, input, input)


@_deterministic_all_reduce_.register_fake
def _deterministic_all_reduce_fake(
    input: torch.Tensor,
    collective_handle: int,
) -> None:
    del input, collective_handle


def deterministic_all_reduce_inplace(
    input: torch.Tensor,
    *,
    collective_handle: int,
) -> torch.Tensor:
    """Run the graph-visible deterministic all-reduce in place."""

    _deterministic_all_reduce_(input, collective_handle)
    return input


class DeterministicCollective:
    """Correctness-first TP-invariant CUDA collectives for one eight-GPU node.

    TP sizes 1, 2, 4, and 8 use nested prefixes of the same balanced tree.
    A reduction is cross-TP bitwise invariant when every rank input is the
    corresponding contiguous subtree root of one canonical finest-grained
    reduction, as produced by a TBIK-compatible row-parallel kernel. Every
    node evaluates the lower logical subtree before the higher one.

    One instance owns a symmetric CUDA IPC staging buffer. All ranks must call
    its methods in the same order with matching shapes and dtypes. Device-side
    IPC sequence fences order staging and payload access without a steady-state
    host barrier and advance correctly during CUDA Graph replay.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before collectives")
        if not torch.cuda.is_available():
            raise RuntimeError("deterministic collectives require CUDA")
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")

        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "deterministic collectives require world_size in "
                f"{_SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )

        if device is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            normalized_device = torch.device("cuda", device)
        else:
            normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError(f"deterministic collectives require a CUDA device, got {device!r}")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        if normalized_device.index != torch.cuda.current_device():
            raise ValueError(
                "the collective device must be the current CUDA device; call "
                f"torch.cuda.set_device({normalized_device.index}) first"
            )

        try:
            from rl_engine import _C
        except ImportError as exc:
            raise RuntimeError(
                "the RL-Kernel CUDA extension is required; rebuild with "
                "`pip install --no-build-isolation -e .`"
            ) from exc
        required_symbols = (
            "deterministic_collective_ipc_meta",
            "deterministic_collective_create",
            "deterministic_collective_destroy",
            "deterministic_collective_stage",
            "deterministic_collective_all_reduce",
            "deterministic_collective_all_reduce_fused",
            "deterministic_collective_reduce_scatter",
            "deterministic_collective_all_gather",
            "deterministic_collective_all_gather_fused",
        )
        missing = [name for name in required_symbols if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                "the RL-Kernel CUDA extension lacks deterministic collectives: "
                + ", ".join(missing)
            )

        self.device = normalized_device
        self.max_size_bytes = int(max_size_bytes)
        self._extension = _C
        self._lock = threading.Lock()
        self._handle = 0
        self._validated_signatures: set[tuple[Any, ...]] = set()
        self._staging = torch.zeros(
            _COLLECTIVE_STAGING_FRAMES * (self.max_size_bytes + _COLLECTIVE_FRAME_METADATA_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )

        handle, offset = self._extension.deterministic_collective_ipc_meta(self._staging)
        local_meta = {
            "handle": handle,
            "offset": int(offset),
            "capacity": self.max_size_bytes,
            "hostname": socket.gethostname(),
        }
        gathered_meta: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_meta, local_meta, group=self.group)
        if any(meta is None for meta in gathered_meta):
            raise RuntimeError("failed to exchange CUDA IPC metadata")
        complete_meta = [meta for meta in gathered_meta if meta is not None]
        hostnames = {meta["hostname"] for meta in complete_meta}
        if len(hostnames) != 1:
            raise ValueError("deterministic collectives require all ranks on one host")
        capacities = {meta["capacity"] for meta in complete_meta}
        if capacities != {self.max_size_bytes}:
            raise ValueError("all ranks must use the same max_size_bytes")

        handles = [meta["handle"] for meta in complete_meta]
        offsets = [meta["offset"] for meta in complete_meta]
        self._handle = self._extension.deterministic_collective_create(
            self._staging,
            handles,
            offsets,
            self.rank,
        )
        self._synchronize_ranks()

    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Return the TBIK-compatible fixed-tree sum on every rank.

        Supported dtypes are float32, float16, and bfloat16. ``out`` may alias
        ``input``; the input is staged before the output kernel starts. Cross-TP
        invariance requires inputs to follow the class-level subtree contract.
        """

        self._check_open()
        self._validate_reduction_input(input)
        if out is None:
            out = torch.empty_like(input)
        self._validate_output(out, input)

        with self._lock:
            if validate_signature:
                self._validate_matching_signature("all_reduce", input)
            self._extension.deterministic_collective_all_reduce_fused(self._handle, input, out)
        return out

    def all_gather(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Gather rank-ordered input bit patterns along dimension 0.

        The result is cross-TP invariant when every TP configuration partitions
        the same global dimension 0 into equal contiguous rank-ordered shards.
        """

        self._check_open()
        self._validate_gather_input(input)
        output_shape = (input.size(0) * self.world_size, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_sharded_output(out, input, output_shape)

        with self._lock:
            if validate_signature:
                self._validate_matching_signature("all_gather", input)
            self._extension.deterministic_collective_all_gather_fused(self._handle, input, out)
        return out

    def all_gather_many(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        validate_signature: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Gather several tensors through the available single-tensor ABI."""

        if not inputs:
            raise ValueError("all_gather_many requires at least one input")
        return tuple(
            self.all_gather(input, validate_signature=validate_signature) for input in inputs
        )

    def reduce_scatter(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """TBIK-compatible fixed-tree sum, then a rank-ordered dimension-0 scatter.

        Concatenating all rank outputs is cross-TP invariant when inputs follow
        the class-level subtree contract and the global dimension-0 size is fixed.
        """

        self._check_open()
        self._validate_reduction_input(input)
        if input.dim() == 0:
            raise ValueError("reduce_scatter input must have at least one dimension")
        if input.size(0) % self.world_size != 0:
            raise ValueError(
                "reduce_scatter input.size(0) must be divisible by "
                f"world_size={self.world_size}; got {input.size(0)}"
            )
        output_shape = (input.size(0) // self.world_size, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_sharded_output(out, input, output_shape)

        with self._lock:
            if validate_signature:
                self._validate_matching_signature("reduce_scatter", input)
            self._extension.deterministic_collective_stage(self._handle, input)
            self._extension.deterministic_collective_reduce_scatter(self._handle, out)
        return out

    def reduce_scatter_many(
        self,
        inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        outs: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        validate_signature: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Compatibility fallback for CUDA IPC collectives.

        The native CUDA IPC backend has no packed transport primitive yet, so
        it preserves its established behavior by issuing the individual
        fixed-tree calls. The ROCm transport subclass overrides this method
        with a packed implementation.
        """

        values = tuple(inputs)
        if not values:
            raise ValueError("reduce_scatter_many requires at least one input")
        if outs is not None and len(outs) != len(values):
            raise ValueError("reduce_scatter_many outs must match the number of inputs")
        results = tuple(
            self.reduce_scatter(
                value,
                out=None if outs is None else outs[index],
                validate_signature=validate_signature,
            )
            for index, value in enumerate(values)
        )
        return results

    def close(self) -> None:
        """Release imported CUDA IPC mappings after the last collective call."""

        handle = getattr(self, "_handle", 0)
        if not handle:
            return
        torch.cuda.synchronize(self.device)
        self._handle = 0
        self._extension.deterministic_collective_destroy(handle)

    def __enter__(self) -> DeterministicCollective:
        self._check_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self) -> None:
        if not getattr(self, "_handle", 0):
            raise RuntimeError("deterministic collective is closed")

    def _validate_reduction_input(self, input: torch.Tensor) -> None:
        if not input.is_cuda or input.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {input.device}")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        if input.dtype not in _REDUCTION_DTYPES:
            raise TypeError(
                "deterministic reductions support float32, float16, and bfloat16; "
                f"got {input.dtype}"
            )
        input_bytes = input.numel() * input.element_size()
        if input_bytes > self.max_size_bytes:
            raise ValueError(
                f"input requires {input_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _validate_gather_input(self, input: torch.Tensor) -> None:
        if not input.is_cuda or input.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {input.device}")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        if input.dim() == 0:
            raise ValueError("all_gather input must have at least one dimension")
        input_bytes = input.numel() * input.element_size()
        if input_bytes > self.max_size_bytes:
            raise ValueError(
                f"input requires {input_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _validate_output(self, output: torch.Tensor, input: torch.Tensor) -> None:
        if output.device != input.device:
            raise ValueError("out must be on the same device as input")
        if output.dtype != input.dtype:
            raise TypeError("out must have the same dtype as input")
        if output.shape != input.shape:
            raise ValueError("out must have the same shape as input")
        if not output.is_contiguous():
            raise ValueError("out must be contiguous")

    def _validate_sharded_output(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_shape: tuple[int, ...],
    ) -> None:
        if output.device != input.device:
            raise ValueError("out must be on the same device as input")
        if output.dtype != input.dtype:
            raise TypeError("out must have the same dtype as input")
        if output.shape != output_shape:
            raise ValueError(f"out must have shape {output_shape}, got {tuple(output.shape)}")
        if not output.is_contiguous():
            raise ValueError("out must be contiguous")

    def _validate_matching_signature(self, op_name: str, input: torch.Tensor) -> None:
        signature = (op_name, tuple(input.shape), str(input.dtype), input.numel())
        if signature in self._validated_signatures:
            return
        signatures: list[tuple[Any, ...] | None] = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(peer_signature != signature for peer_signature in signatures):
            raise ValueError(
                f"all ranks must call {op_name} with matching shapes and dtypes; got {signatures}"
            )
        self._validated_signatures.add(signature)

    def _validate_matching_many_signature(
        self,
        op_name: str,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        signature = (
            op_name,
            tuple((tuple(input.shape), str(input.dtype), input.numel()) for input in inputs),
        )
        if signature in self._validated_signatures:
            return
        signatures: list[tuple[Any, ...] | None] = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(peer_signature != signature for peer_signature in signatures):
            raise ValueError(
                f"all ranks must call {op_name} with matching tensors; got {signatures}"
            )
        self._validated_signatures.add(signature)

    def _validate_many_capacity(self, inputs: tuple[torch.Tensor, ...]) -> None:
        total_bytes = 0
        for input in inputs:
            total_bytes = (total_bytes + 15) & ~15
            total_bytes += input.numel() * input.element_size()
        if total_bytes > self.max_size_bytes:
            raise ValueError(
                f"inputs require {total_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _synchronize_ranks(self) -> None:
        torch.cuda.synchronize(self.device)
        backend = dist.get_backend(self.group)
        if backend == dist.Backend.NCCL or str(backend).lower() == "nccl":
            dist.barrier(group=self.group, device_ids=[self.device.index])
        else:
            dist.barrier(group=self.group)


class TorchDistributedDeterministicCollective:
    """Correctness-first collectives using AllGather as transport only.

    Rank inputs are gathered without arithmetic and reduced locally as the
    balanced tree ``((rank0 + rank1) + (rank2 + rank3)) + ...``. Consequently,
    all ranks execute the exact same floating-point expression. TP sizes 1,
    2, 4, and 8 are nested prefixes of that expression and match the existing
    CUDA IPC collective's ordering.

    The generic class also supports a CPU/Gloo process group, which is useful
    as an executable reference. Production ROCm callers should use
    :class:`RCCLDeterministicCollective` or
    :func:`create_deterministic_collective` so backend validation fails closed.
    All ranks must call methods in the same order with matching input shapes
    and dtypes, and construct the instance with the same ``max_size_bytes``.
    """

    backend_id = "torch_distributed_balanced_tree"
    transport_only = True
    reduction_order = "balanced_rank_tree"
    supports_async_overlap = False
    supports_compute_communication_fusion = False

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before collectives")
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")

        self.group = group if group is not None else dist.group.WORLD
        self.rank = int(dist.get_rank(group=self.group))
        self.world_size = int(dist.get_world_size(group=self.group))
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "deterministic collectives require world_size in "
                f"{_SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )

        self.device = self._normalize_device(device)
        self.max_size_bytes = int(max_size_bytes)
        self._backend = str(dist.get_backend(self.group)).lower()
        self._lock = threading.Lock()
        self._closed = False
        # Keep a lifecycle marker for callers that historically inspected the
        # CUDA IPC collective's ``_handle`` while managing the cache. Concrete
        # transports own any native resource through their own state.
        self._handle = id(self)
        # One dtype-agnostic byte workspace is grown on demand and reused by
        # reduction collectives. AllGather writes directly into its output.
        self._workspace: torch.Tensor | None = None
        # A Python-object collective is useful for catching a mismatched new
        # signature, but running one on every hot-path call dominates small
        # message latency. Validate each local signature once and then rely on
        # the standard collective contract that ranks call operations in the
        # same order.
        self._validated_signatures: set[tuple[Any, ...]] = set()
        self._validate_matching_capacity()

    @staticmethod
    def _normalize_device(
        device: torch.device | str | int | None,
    ) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                normalized = torch.device("cuda", torch.cuda.current_device())
            else:
                normalized = torch.device("cpu")
        elif isinstance(device, int):
            normalized = torch.device("cuda", device)
        else:
            normalized = torch.device(device)

        if normalized.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("a CUDA/ROCm device was requested but none is available")
            current_device = torch.cuda.current_device()
            if normalized.index is None:
                normalized = torch.device("cuda", current_device)
            if normalized.index != current_device:
                raise ValueError(
                    "the collective device must be the current CUDA/ROCm device; call "
                    f"torch.cuda.set_device({normalized.index}) first"
                )
        return normalized

    @property
    def closed(self) -> bool:
        """Whether this instance rejects further collective calls."""

        return self._closed

    @property
    def workspace_size_bytes(self) -> int:
        """Currently retained reduction workspace size in bytes."""

        workspace = self._workspace
        return 0 if workspace is None else int(workspace.numel())

    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Return the fixed balanced-tree sum on every rank."""

        self._check_open()
        self._validate_reduction_input(input)
        if out is None:
            out = torch.empty_like(input)
        self._validate_output(out, input, tuple(input.shape))
        if self.world_size == 1:
            out.copy_(input)
            return out

        with self._lock:
            self._check_open()
            if validate_signature:
                self._validate_matching_signature("all_reduce", input)
            if self._direct_all_reduce(input, out):
                return out
            rank_inputs = self._all_gather_transport(input)
            if not self._fused_reduction(
                rank_inputs,
                out,
                operation="all_reduce",
            ):
                reduced = self._balanced_tree_sum(rank_inputs)
                out.copy_(reduced)
        return out

    def all_gather(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Gather rank-ordered input bit patterns along dimension 0."""

        self._check_open()
        self._validate_gather_input(input)
        output_shape = (input.size(0) * self.world_size, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_output(out, input, output_shape)
        if self.world_size == 1:
            out.copy_(input)
            return out

        with self._lock:
            self._check_open()
            if validate_signature:
                self._validate_matching_signature("all_gather", input)
            if self._direct_all_gather(input, out):
                return out
            self._all_gather_transport(input, gathered_flat=out.view(-1))
        return out

    def all_gather_many(
        self,
        inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        validate_signature: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Gather several tensors through the platform transport."""

        values = tuple(inputs)
        if not values:
            raise ValueError("all_gather_many requires at least one input")
        return tuple(
            self.all_gather(value, validate_signature=validate_signature)
            for value in values
        )

    def reduce_scatter(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        validate_signature: bool = True,
    ) -> torch.Tensor:
        """Fixed-tree sum followed by rank-ordered dimension-0 slicing."""

        self._check_open()
        self._validate_reduction_input(input)
        if input.dim() == 0:
            raise ValueError("reduce_scatter input must have at least one dimension")
        if input.size(0) % self.world_size != 0:
            raise ValueError(
                "reduce_scatter input.size(0) must be divisible by "
                f"world_size={self.world_size}; got {input.size(0)}"
            )
        rows_per_rank = input.size(0) // self.world_size
        output_shape = (rows_per_rank, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_output(out, input, output_shape)
        if self.world_size == 1:
            out.copy_(input)
            return out

        with self._lock:
            self._check_open()
            if validate_signature:
                self._validate_matching_signature("reduce_scatter", input)
            if self._direct_reduce_scatter(input, out):
                return out
            rank_inputs = self._all_gather_transport(input)
            begin = self.rank * rows_per_rank
            # Only this rank's output shard participates in the reduction. The
            # previous implementation reduced every global row and sliced the
            # result afterwards, doing world_size times more arithmetic than
            # ReduceScatter needs. The fixed rank tree is unchanged.
            reduced = rank_inputs[:, begin : begin + rows_per_rank]
            if not self._fused_reduction(reduced, out, operation="reduce_scatter"):
                reduced = self._balanced_tree_sum(reduced)
                out.copy_(reduced)
        return out

    def reduce_scatter_many(
        self,
        inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        outs: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        validate_signature: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        """Reduce-scatter independent tensors in one fixed-tree collective.

        The tensors are packed along their final dimension, so each tensor's
        element still follows the same balanced rank tree as an individual
        ``reduce_scatter`` call. This is useful for independent gradient lanes:
        packing them together removes one RCCL launch without changing the
        floating-point expression for either lane. Inputs must have matching
        shape/device/dtype except for the final dimension.
        """

        self._check_open()
        values = tuple(inputs)
        if not values:
            raise ValueError("reduce_scatter_many requires at least one input")
        if outs is not None and len(outs) != len(values):
            raise ValueError("reduce_scatter_many outs must match the number of inputs")
        if len(values) == 1:
            return (
                self.reduce_scatter(
                    values[0],
                    out=None if outs is None else outs[0],
                    validate_signature=validate_signature,
                ),
            )

        first = values[0]
        self._validate_reduction_input(first)
        if first.dim() < 2:
            raise ValueError(
                "reduce_scatter_many inputs must have at least two dimensions "
                "when packing independent lanes"
            )
        if first.size(0) % self.world_size != 0:
            raise ValueError("reduce_scatter_many inputs must have a divisible leading dimension")
        for value in values[1:]:
            self._validate_reduction_input(value)
            if value.dim() != first.dim() or value.shape[:-1] != first.shape[:-1]:
                raise ValueError(
                    "reduce_scatter_many inputs must match in rank and all dimensions "
                    "except the final dimension"
                )
            if value.device != first.device or value.dtype != first.dtype:
                raise ValueError("reduce_scatter_many inputs must share device and dtype")
        lane_sizes = tuple(int(value.size(-1)) for value in values)
        rows_per_rank = first.size(0) // self.world_size
        output_shape = (rows_per_rank, *first.shape[1:-1])
        if outs is not None:
            for lane_size, out in zip(lane_sizes, outs, strict=True):
                self._validate_output(
                    out,
                    first,
                    (*output_shape, lane_size),
                )

        packed_bytes = sum(value.numel() * value.element_size() for value in values)
        if self._can_direct_reduce_scatter_many():
            if packed_bytes > self.max_size_bytes:
                raise ValueError(
                    "reduce_scatter_many packed input requires "
                    f"{packed_bytes} bytes but max_size_bytes={self.max_size_bytes}"
                )
            direct_outputs = tuple(
                (
                    outs[index]
                    if outs is not None
                    else torch.empty(
                        (*output_shape, lane_size),
                        dtype=first.dtype,
                        device=first.device,
                    )
                )
                for index, lane_size in enumerate(lane_sizes)
            )
            with self._lock:
                self._check_open()
                if validate_signature:
                    self._validate_matching_signature(
                        f"reduce_scatter_many:{lane_sizes}",
                        first,
                    )
                if self._direct_reduce_scatter_many(values, direct_outputs):
                    return direct_outputs

        if packed_bytes > _PACKED_REDUCE_SCATTER_MAX_BYTES:
            # A single packed AllGather moves the same bytes as two separate
            # calls but loses RCCL's smaller-message algorithm. Use the
            # established per-lane path above the measured crossover; this
            # keeps the convenience API from regressing large FFN gradients.
            return tuple(
                self.reduce_scatter(
                    value,
                    out=None if outs is None else outs[index],
                    validate_signature=validate_signature,
                )
                for index, value in enumerate(values)
            )

        if packed_bytes > self.max_size_bytes:
            raise ValueError(
                "reduce_scatter_many packed input requires "
                f"{packed_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )
        packed = torch.cat(values, dim=-1)
        packed_out = torch.empty(
            (packed.size(0) // self.world_size, *packed.shape[1:]),
            dtype=packed.dtype,
            device=packed.device,
        )
        with self._lock:
            self._check_open()
            # Include lane boundaries in the signature. Equal packed shapes
            # alone do not guarantee that every rank will split the result the
            # same way, which could silently associate gradients with the
            # wrong lane.
            if validate_signature:
                self._validate_matching_signature(
                    f"reduce_scatter_many:{lane_sizes}",
                    packed,
                )
            if self._direct_reduce_scatter(packed, packed_out):
                pieces = tuple(packed_out.split(tuple(value.size(-1) for value in values), dim=-1))
                if outs is None:
                    return pieces
                result: list[torch.Tensor] = []
                for piece, out in zip(pieces, outs, strict=True):
                    out.copy_(piece)
                    result.append(out)
                return tuple(result)
            rank_inputs = self._all_gather_transport(packed)
            begin = self.rank * rows_per_rank
            reduced = rank_inputs[:, begin : begin + rows_per_rank]
            if not self._fused_reduction(
                reduced,
                packed_out,
                operation="reduce_scatter",
            ):
                reduced = self._balanced_tree_sum(reduced)
                packed_out.copy_(reduced)

        pieces = tuple(packed_out.split(tuple(value.size(-1) for value in values), dim=-1))
        if outs is None:
            return pieces
        result: list[torch.Tensor] = []
        for piece, out in zip(pieces, outs, strict=True):
            out.copy_(piece)
            result.append(out)
        return tuple(result)

    def close(self) -> None:
        """Close the instance.

        Closing releases the lazily allocated reduction workspace and marks
        the lifecycle boundary. Collective calls are blocking at this API.
        """

        with self._lock:
            self._workspace = None
            self._validated_signatures.clear()
            self._closed = True
            self._handle = 0

    def __enter__(self) -> TorchDistributedDeterministicCollective:
        self._check_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self) -> None:
        if getattr(self, "_closed", True):
            raise RuntimeError("deterministic collective is closed")

    def _validate_tensor(self, input: torch.Tensor) -> None:
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"input must be a torch.Tensor, got {type(input)!r}")
        if input.device != self.device:
            raise ValueError(f"input must be on {self.device}, got {input.device}")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        input_bytes = input.numel() * input.element_size()
        if input_bytes > self.max_size_bytes:
            raise ValueError(
                f"input requires {input_bytes} bytes but max_size_bytes={self.max_size_bytes}"
            )

    def _validate_reduction_input(self, input: torch.Tensor) -> None:
        self._validate_tensor(input)
        if input.dtype not in _REDUCTION_DTYPES:
            raise TypeError(
                "deterministic reductions support float32, float16, and bfloat16; "
                f"got {input.dtype}"
            )

    def _validate_gather_input(self, input: torch.Tensor) -> None:
        self._validate_tensor(input)
        if input.dim() == 0:
            raise ValueError("all_gather input must have at least one dimension")

    @staticmethod
    def _validate_output(
        output: torch.Tensor,
        input: torch.Tensor,
        output_shape: tuple[int, ...],
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"out must be a torch.Tensor, got {type(output)!r}")
        if output.device != input.device:
            raise ValueError("out must be on the same device as input")
        if output.dtype != input.dtype:
            raise TypeError("out must have the same dtype as input")
        if tuple(output.shape) != output_shape:
            raise ValueError(f"out must have shape {output_shape}, got {tuple(output.shape)}")
        if not output.is_contiguous():
            raise ValueError("out must be contiguous")

    def _validate_matching_signature(self, op_name: str, input: torch.Tensor) -> None:
        if self.world_size == 1:
            return
        signature = (op_name, tuple(input.shape), str(input.dtype), input.numel())
        if signature in self._validated_signatures:
            return
        signatures: list[tuple[Any, ...] | None] = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(peer_signature != signature for peer_signature in signatures):
            raise ValueError(
                f"all ranks must call {op_name} with matching shapes and dtypes; got {signatures}"
            )
        self._validated_signatures.add(signature)

    def _validate_matching_capacity(self) -> None:
        if self.world_size == 1:
            return
        capacities: list[int | None] = [None] * self.world_size
        dist.all_gather_object(capacities, self.max_size_bytes, group=self.group)
        if any(peer_capacity != self.max_size_bytes for peer_capacity in capacities):
            raise ValueError(f"all ranks must use the same max_size_bytes; got {capacities}")

    def _all_gather_transport(
        self,
        input: torch.Tensor,
        *,
        gathered_flat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Flattening makes the output contract independent of whether a given
        # ProcessGroup implements the concatenation or stacking form of AG.
        if self.world_size == 1:
            if gathered_flat is None:
                gathered_flat = input.clone().view(-1)
            else:
                gathered_flat.copy_(input.view(-1))
            return gathered_flat.reshape((1, *input.shape))
        input_flat = input.view(-1)
        required_elements = self.world_size * input_flat.numel()
        if gathered_flat is None:
            gathered_flat = self._workspace_for(input, required_elements)
        elif (
            gathered_flat.numel() != required_elements
            or gathered_flat.dtype != input.dtype
            or gathered_flat.device != input.device
            or not gathered_flat.is_contiguous()
        ):
            raise ValueError("gathered transport output has an invalid layout")
        if "nccl" in self._backend:
            # PyTorch exposes RCCL through the NCCL ProcessGroup API. Keep this
            # as a tensor-only transport; reduction happens below.
            dist.all_gather_into_tensor(gathered_flat, input_flat, group=self.group)
        else:
            # Some reference backends (notably Gloo versions without
            # all_gather_into_tensor) only implement the list API.
            gathered_chunks = list(
                gathered_flat.reshape(self.world_size, input_flat.numel()).unbind(0)
            )
            dist.all_gather(gathered_chunks, input_flat, group=self.group)
        return gathered_flat.reshape((self.world_size, *input.shape))

    def _workspace_for(self, input: torch.Tensor, required_elements: int) -> torch.Tensor:
        required_bytes = required_elements * input.element_size()
        workspace = self._workspace
        if workspace is None or workspace.numel() < required_bytes:
            workspace = torch.empty(required_bytes, dtype=torch.uint8, device=self.device)
            self._workspace = workspace
        # Tensor.view(dtype) reinterprets the aligned byte allocation without
        # an allocation or copy. Restrict the view to the current operation.
        return workspace[:required_bytes].view(input.dtype)

    def _direct_all_reduce(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        return False

    def _direct_reduce_scatter(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        return False

    def _can_direct_reduce_scatter_many(self) -> bool:
        return False

    def _direct_reduce_scatter_many(
        self,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> bool:
        return False

    def _direct_all_gather(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        return False

    @staticmethod
    def _balanced_tree_sum(rank_inputs: torch.Tensor) -> torch.Tensor:
        world_size = rank_inputs.size(0)
        if world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "balanced reduction requires rank inputs for world_size in "
                f"{_SUPPORTED_WORLD_SIZES}, got {world_size}"
            )
        # ``rank_inputs`` is the private transport workspace for reductions,
        # so fixed-tree nodes can be accumulated in place. This preserves the
        # exact pairings while avoiding one temporary allocation per tree node.
        stride = 1
        while stride < world_size:
            for index in range(0, world_size, 2 * stride):
                rank_inputs[index].add_(rank_inputs[index + stride])
            stride *= 2
        return rank_inputs[0]

    @staticmethod
    def _fused_reduction(
        rank_inputs: torch.Tensor,
        output: torch.Tensor,
        *,
        operation: str,
    ) -> bool:
        """Use the optional ROCm fused fixed-tree kernel when available.

        The extension is deliberately optional: CPU/Gloo reference collectives
        and installations built without the ROCm kernel retain the executable
        Python implementation above.
        """

        if getattr(torch.version, "hip", None) is None or not rank_inputs.is_cuda:
            return False
        try:
            from rl_engine import _C
        except ImportError:
            return False
        if operation == "all_reduce":
            fn = getattr(_C, "deterministic_collective_rocm_all_reduce", None)
            if fn is not None:
                fn(rank_inputs, output)
                return True
        elif operation == "reduce_scatter":
            fn = getattr(_C, "deterministic_collective_rocm_reduce_scatter", None)
            if fn is not None:
                fn(rank_inputs, output)
                return True
        return False


class RCCLDeterministicCollective(TorchDistributedDeterministicCollective):
    """Single-node ROCm fixed-tree collective using HIP IPC and RCCL."""

    backend_id = "rocm_ipc_fixed_tree"

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        device: torch.device | str | int | None = None,
        *,
        max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if getattr(torch.version, "hip", None) is None:
            raise RuntimeError("RCCL deterministic collectives require a ROCm PyTorch build")
        if not torch.cuda.is_available():
            raise RuntimeError("RCCL deterministic collectives require an available ROCm device")
        if device is not None and not isinstance(device, int):
            requested_device = torch.device(device)
            if requested_device.type != "cuda":
                raise ValueError(
                    f"RCCL deterministic collectives require a ROCm device, got {device!r}"
                )
        super().__init__(group=group, device=device, max_size_bytes=max_size_bytes)
        if self.device.type != "cuda":
            raise ValueError(
                f"RCCL deterministic collectives require a ROCm device, got {device!r}"
            )
        if "nccl" not in self._backend:
            raise RuntimeError(
                "RCCL deterministic collectives require PyTorch's NCCL process-group API"
            )
        self._ipc_handle = 0
        self._ipc_staging: torch.Tensor | None = None
        self._initialize_ipc_transport()

    @property
    def workspace_size_bytes(self) -> int:
        staging = self._ipc_staging
        staging_bytes = 0 if staging is None else int(staging.numel())
        return staging_bytes + super().workspace_size_bytes

    def _initialize_ipc_transport(self) -> None:
        if self.world_size == 1:
            return
        try:
            from rl_engine import _C
        except ImportError:
            return
        required_symbols = (
            "deterministic_collective_rocm_ipc_allocate",
            "deterministic_collective_rocm_ipc_meta",
            "deterministic_collective_rocm_ipc_create",
            "deterministic_collective_rocm_ipc_synchronize",
            "deterministic_collective_rocm_ipc_destroy",
            "deterministic_collective_rocm_ipc_stage",
            "deterministic_collective_rocm_ipc_all_reduce",
            "deterministic_collective_rocm_ipc_all_reduce_input",
            "deterministic_collective_rocm_ipc_reduce_scatter",
            "deterministic_collective_rocm_ipc_reduce_scatter_input",
            "deterministic_collective_rocm_ipc_reduce_scatter_many",
            "deterministic_collective_rocm_ipc_all_gather",
            "deterministic_collective_rocm_ipc_all_gather_input",
        )
        if any(not hasattr(_C, symbol) for symbol in required_symbols):
            return

        staging = _C.deterministic_collective_rocm_ipc_allocate(self.max_size_bytes)
        handle, offset = _C.deterministic_collective_rocm_ipc_meta(staging)
        local_metadata = (socket.gethostname(), handle, int(offset))
        gathered_metadata: list[tuple[str, list[int], int] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_metadata, local_metadata, group=self.group)
        if any(metadata is None for metadata in gathered_metadata):
            raise RuntimeError("failed to exchange ROCm IPC metadata")
        complete_metadata = [metadata for metadata in gathered_metadata if metadata is not None]
        if len({metadata[0] for metadata in complete_metadata}) != 1:
            return
        self._ipc_handle = int(
            _C.deterministic_collective_rocm_ipc_create(
                staging,
                [metadata[1] for metadata in complete_metadata],
                [metadata[2] for metadata in complete_metadata],
                self.rank,
            )
        )
        self._ipc_staging = staging

    def _direct_all_reduce(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        handle = self._ipc_handle
        if not handle:
            return False
        from rl_engine import _C

        input_bytes = input.numel() * input.element_size()
        if (
            _ROCM_IPC_DIRECT_ALL_REDUCE_MAX_BYTES
            < input_bytes
            < _ROCM_IPC_SHARDED_ALL_REDUCE_MIN_BYTES
            and input.numel() % self.world_size == 0
        ):
            return False

        if (
            input_bytes <= _ROCM_IPC_DIRECT_ALL_REDUCE_MAX_BYTES
            or input.numel() % self.world_size != 0
        ):
            _C.deterministic_collective_rocm_ipc_all_reduce_input(
                handle,
                input,
                output,
            )
            return True

        shard = self._workspace_for(input, input.numel() // self.world_size)
        _C.deterministic_collective_rocm_ipc_reduce_scatter_input(
            handle,
            input,
            shard,
        )
        dist.all_gather_into_tensor(output.view(-1), shard, group=self.group)
        return True

    def _direct_reduce_scatter(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        handle = self._ipc_handle
        if not handle:
            return False
        from rl_engine import _C

        _C.deterministic_collective_rocm_ipc_reduce_scatter_input(
            handle,
            input,
            output,
        )
        return True

    def _can_direct_reduce_scatter_many(self) -> bool:
        return bool(self._ipc_handle)

    def _direct_reduce_scatter_many(
        self,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
    ) -> bool:
        handle = self._ipc_handle
        if not handle:
            return False
        from rl_engine import _C

        _C.deterministic_collective_rocm_ipc_reduce_scatter_many(
            handle,
            inputs,
            outputs,
        )
        return True

    def _direct_all_gather(self, input: torch.Tensor, output: torch.Tensor) -> bool:
        handle = self._ipc_handle
        input_bytes = input.numel() * input.element_size()
        if not handle or input_bytes > _ROCM_IPC_ALL_GATHER_MAX_BYTES:
            return False
        from rl_engine import _C

        _C.deterministic_collective_rocm_ipc_all_gather_input(
            handle,
            input,
            output,
        )
        return True

    def close(self) -> None:
        handle = getattr(self, "_ipc_handle", 0)
        if handle:
            from rl_engine import _C

            _C.deterministic_collective_rocm_ipc_synchronize(handle)
            torch.cuda.synchronize(self.device)
            self._ipc_handle = 0
            _C.deterministic_collective_rocm_ipc_destroy(handle)
            self._ipc_staging = None
        super().close()


def create_deterministic_collective(
    group: dist.ProcessGroup | None = None,
    device: torch.device | str | int | None = None,
    *,
    max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
) -> Any:
    """Create the platform-appropriate deterministic collective.

    CUDA uses the native ``DeterministicCollective`` implementation. ROCm uses
    HIP IPC or RCCL for rank-ordered transport while preserving the fixed local
    reduction tree. The returned object has independent ownership. Shared caches
    may replace an entry without closing it immediately because active autograd
    contexts can retain the previous instance until their work completes.
    """

    if getattr(torch.version, "hip", None) is not None:
        return RCCLDeterministicCollective(
            group=group,
            device=device,
            max_size_bytes=max_size_bytes,
        )

    return DeterministicCollective(
        group=group,
        device=device,
        max_size_bytes=max_size_bytes,
    )


def collective_for_group(
    group: dist.ProcessGroup | None,
    *,
    min_size_bytes: int = 0,
    minimum_capacity_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
    device: torch.device | str | int | None = None,
) -> Any | None:
    """Return the process-local platform collective shared by hot-path ops."""

    if group is None:
        return None
    if min_size_bytes < 0:
        raise ValueError("min_size_bytes must be non-negative")
    if minimum_capacity_bytes <= 0:
        raise ValueError("minimum_capacity_bytes must be positive")

    rank = int(dist.get_rank(group=group))
    world_size = int(dist.get_world_size(group=group))
    if device is None:
        device_index = torch.cuda.current_device()
    else:
        normalized_device = (
            torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        )
        device_index = (
            torch.cuda.current_device()
            if normalized_device.index is None
            else normalized_device.index
        )
    key = (id(group), rank, world_size, device_index)
    cached = _COLLECTIVES.get(key)
    if cached is not None and cached.max_size_bytes >= min_size_bytes:
        return cached
    # Borrowers such as Attention autograd contexts can outlive this cache
    # entry. Replacing an undersized entry must not invalidate those live
    # references; normal Python ownership closes it after the last borrower.

    collective = create_deterministic_collective(
        group=group,
        device=device_index,
        max_size_bytes=max(minimum_capacity_bytes, min_size_bytes),
    )
    _COLLECTIVES[key] = collective
    return collective


__all__ = [
    "DETERMINISTIC_ALL_REDUCE_OP",
    "DeterministicCollective",
    "RCCLDeterministicCollective",
    "TorchDistributedDeterministicCollective",
    "collective_for_group",
    "create_deterministic_collective",
    "deterministic_all_reduce_inplace",
]
