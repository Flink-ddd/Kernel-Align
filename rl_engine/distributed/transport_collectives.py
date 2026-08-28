# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic reductions built on rank-ordered tensor transport.

The transport in this module never performs a floating-point reduction.  It
only gathers every rank's input, after which each rank evaluates the same
balanced reduction tree locally.  In a ROCm build, PyTorch's ``nccl`` backend
is RCCL and therefore ``all_gather_into_tensor`` provides the transport.
"""

from __future__ import annotations

import threading
from types import TracebackType
from typing import Any

import torch
import torch.distributed as dist

_SUPPORTED_WORLD_SIZES = (1, 2, 4, 8)
_DEFAULT_MAX_SIZE_BYTES = 64 * 1024 * 1024
_REDUCTION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class TorchDistributedDeterministicCollective:
    """Correctness-first collectives using AllGather as transport only.

    Rank inputs are gathered without arithmetic and reduced locally as the
    balanced tree ``((rank0 + rank1) + (rank2 + rank3)) + ...``.  Consequently,
    all ranks execute the exact same floating-point expression.  TP sizes 1,
    2, 4, and 8 are nested prefixes of that expression and match the existing
    CUDA IPC collective's ordering.

    The generic class also supports a CPU/Gloo process group, which is useful
    as an executable reference.  Production ROCm callers should use
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
    ) -> torch.Tensor:
        """Return the fixed balanced-tree sum on every rank."""

        self._check_open()
        self._validate_reduction_input(input)
        if out is None:
            out = torch.empty_like(input)
        self._validate_output(out, input, tuple(input.shape))

        with self._lock:
            self._check_open()
            self._validate_matching_signature("all_reduce", input)
            rank_inputs = self._all_gather_transport(input)
            reduced = self._balanced_tree_sum(rank_inputs)
            out.copy_(reduced)
        return out

    def all_gather(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather rank-ordered input bit patterns along dimension 0."""

        self._check_open()
        self._validate_gather_input(input)
        output_shape = (input.size(0) * self.world_size, *input.shape[1:])
        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        self._validate_output(out, input, output_shape)

        with self._lock:
            self._check_open()
            self._validate_matching_signature("all_gather", input)
            self._all_gather_transport(input, gathered_flat=out.view(-1))
        return out

    def reduce_scatter(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
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

        with self._lock:
            self._check_open()
            self._validate_matching_signature("reduce_scatter", input)
            rank_inputs = self._all_gather_transport(input)
            reduced = self._balanced_tree_sum(rank_inputs)
            begin = self.rank * rows_per_rank
            out.copy_(reduced.narrow(0, begin, rows_per_rank))
        return out

    def close(self) -> None:
        """Close the instance.

        Closing releases the lazily allocated reduction workspace and marks
        the lifecycle boundary. Collective calls are blocking at this API.
        """

        with self._lock:
            self._workspace = None
            self._validated_signatures.clear()
            self._closed = True

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
            # PyTorch exposes RCCL through the NCCL ProcessGroup API.  Keep
            # this as a tensor-only transport; reduction happens below.
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


class RCCLDeterministicCollective(TorchDistributedDeterministicCollective):
    """ROCm collective using RCCL AllGather strictly as tensor transport."""

    backend_id = "rccl_all_gather_balanced_tree"

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


def create_deterministic_collective(
    group: dist.ProcessGroup | None = None,
    device: torch.device | str | int | None = None,
    *,
    max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES,
) -> Any:
    """Create the platform-appropriate deterministic collective.

    CUDA keeps using the existing IPC implementation.  ROCm uses RCCL only to
    gather rank inputs, followed by the shared local balanced-tree reduction.
    The returned object has independent ownership; callers may cache it by
    process-group/rank/device and must close an entry before replacing it.
    """

    if getattr(torch.version, "hip", None) is not None:
        return RCCLDeterministicCollective(
            group=group,
            device=device,
            max_size_bytes=max_size_bytes,
        )

    # Import lazily to keep the existing CUDA implementation and its extension
    # checks independent of the generic transport reference above.
    from rl_engine.distributed.collectives import DeterministicCollective

    return DeterministicCollective(
        group=group,
        device=device,
        max_size_bytes=max_size_bytes,
    )


__all__ = [
    "RCCLDeterministicCollective",
    "TorchDistributedDeterministicCollective",
    "create_deterministic_collective",
]
