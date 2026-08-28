# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import socket
import threading
from types import TracebackType
from typing import Any

import torch
import torch.distributed as dist

_SUPPORTED_WORLD_SIZES = (1, 2, 4, 8)
_DEFAULT_MAX_SIZE_BYTES = 64 * 1024 * 1024
_REDUCTION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _npu_available() -> bool:
    """torch.npu only exists after torch_npu is imported; probe defensively."""
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


class DeterministicCollective:
    """Correctness-first TP-invariant collectives for one eight-device node.

    TP sizes 1, 2, 4, and 8 use nested prefixes of the same balanced tree.
    A reduction is cross-TP bitwise invariant when every rank input is the
    corresponding contiguous subtree root of one canonical finest-grained
    reduction, as produced by a TBIK-compatible row-parallel kernel. Every
    node evaluates the lower logical subtree before the higher one.

    CUDA: one instance owns a symmetric CUDA IPC staging buffer; the reduction
    kernel reads every peer's staged data directly. Ascend NPU: no device IPC
    exists, so each reduction first gathers every rank's staged input with an
    HCCL all_gather (bitwise-exact data movement) and then applies the same
    fixed-tree kernel locally -- the reduction order never depends on the HCCL
    algorithm. All ranks must call the methods in the same order with matching
    shapes and dtypes. Calls are host-synchronizing by design; the first
    version prioritizes determinism and lifetime safety over overlap or
    throughput.
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
        self.max_size_bytes = int(max_size_bytes)

        is_cuda = torch.cuda.is_available()
        is_npu = _npu_available()
        if is_cuda:
            self._backend = "cuda"
            normalized_device = torch.device("cuda", torch.cuda.current_device())
            if device is not None:
                normalized_device = (
                    torch.device("cuda", device)
                    if isinstance(device, int)
                    else torch.device(device)
                )
                if normalized_device.type != "cuda":
                    raise ValueError(
                        f"deterministic collectives require a CUDA device, got {device!r}"
                    )
                if normalized_device.index is None:
                    normalized_device = torch.device("cuda", torch.cuda.current_device())
                if normalized_device.index != torch.cuda.current_device():
                    raise ValueError(
                        "the collective device must be the current CUDA device; call "
                        f"torch.cuda.set_device({normalized_device.index}) first"
                    )
            self._load_cuda_extension()
            self.device = normalized_device
            self._create_cuda_state()
        elif is_npu:
            self._backend = "npu"
            normalized_device = torch.device("npu", torch.npu.current_device())
            if device is not None:
                normalized_device = (
                    torch.device("npu", device) if isinstance(device, int) else torch.device(device)
                )
                if normalized_device.type != "npu":
                    raise ValueError(
                        f"deterministic collectives require an NPU device, got {device!r}"
                    )
                if normalized_device.index is None:
                    normalized_device = torch.device("npu", torch.npu.current_device())
                if normalized_device.index != torch.npu.current_device():
                    raise ValueError(
                        "the collective device must be the current NPU device; call "
                        f"torch.npu.set_device({normalized_device.index}) first"
                    )
            self._load_npu_extension()
            self.device = normalized_device
            self._create_npu_state()
        else:
            raise RuntimeError("deterministic collectives require CUDA or Ascend NPU devices")

        self._synchronize_ranks()

    # ------------------------------------------------------------------ #
    # Backend setup
    # ------------------------------------------------------------------ #

    def _load_cuda_extension(self) -> None:
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
            "deterministic_collective_reduce_scatter",
            "deterministic_collective_all_gather",
        )
        missing = [name for name in required_symbols if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                "the RL-Kernel CUDA extension lacks deterministic collectives: "
                + ", ".join(missing)
            )
        self._extension = _C

    def _load_npu_extension(self) -> None:
        try:
            from rl_engine import _C_npu
        except ImportError as exc:
            raise RuntimeError(
                "the RL-Kernel Ascend extension is required; rebuild with "
                "KERNEL_ALIGN_FORCE_ASCEND=1 and `pip install --no-build-isolation -e .`"
            ) from exc
        required_symbols = (
            "deterministic_collective_create",
            "deterministic_collective_destroy",
            "deterministic_collective_stage",
            "deterministic_collective_reduce",
        )
        missing = [name for name in required_symbols if not hasattr(_C_npu, name)]
        if missing:
            raise RuntimeError(
                "the RL-Kernel Ascend extension lacks deterministic collectives: "
                + ", ".join(missing)
            )
        self._extension = _C_npu

    def _create_cuda_state(self) -> None:
        self._lock = threading.Lock()
        self._handle = 0
        self._staging = torch.empty(
            self.max_size_bytes,
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
        gathered_meta = self._exchange_meta(local_meta)
        self._validate_meta(gathered_meta)

        handles = [meta["handle"] for meta in gathered_meta]
        offsets = [meta["offset"] for meta in gathered_meta]
        self._handle = self._extension.deterministic_collective_create(
            self._staging,
            handles,
            offsets,
            self.rank,
        )

    def _create_npu_state(self) -> None:
        self._lock = threading.Lock()
        self._handle = 0
        self._staging = torch.empty(
            self.max_size_bytes,
            dtype=torch.uint8,
            device=self.device,
        )

        # No device IPC on Ascend: only the capacity / hostname invariants are
        # exchanged (the data itself moves through HCCL all_gather per call).
        local_meta = {
            "capacity": self.max_size_bytes,
            "hostname": socket.gethostname(),
        }
        gathered_meta = self._exchange_meta(local_meta)
        self._validate_meta(gathered_meta)

        self._handle = self._extension.deterministic_collective_create(
            self._staging,
            self.world_size,
            self.rank,
        )

    def _exchange_meta(self, local_meta: dict[str, Any]) -> list[dict[str, Any]]:
        gathered_meta: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(gathered_meta, local_meta, group=self.group)
        if any(meta is None for meta in gathered_meta):
            raise RuntimeError("failed to exchange collective metadata")
        return [meta for meta in gathered_meta if meta is not None]

    def _validate_meta(self, complete_meta: list[dict[str, Any]]) -> None:
        hostnames = {meta["hostname"] for meta in complete_meta}
        if len(hostnames) != 1:
            raise ValueError("deterministic collectives require all ranks on one host")
        capacities = {meta["capacity"] for meta in complete_meta}
        if capacities != {self.max_size_bytes}:
            raise ValueError("all ranks must use the same max_size_bytes")

    # ------------------------------------------------------------------ #
    # Public collectives
    # ------------------------------------------------------------------ #

    def all_reduce(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the TBIK-compatible fixed-tree sum on every rank.

        Supported dtypes are float32, float16, and bfloat16. ``out`` may alias
        ``input``; the input is staged before the reduction kernel starts.
        Cross-TP invariance requires inputs to follow the class-level subtree
        contract.
        """

        self._check_open()
        self._validate_reduction_input(input)
        if out is None:
            out = torch.empty_like(input)
        self._validate_output(out, input)

        with self._lock:
            self._validate_matching_signature("all_reduce", input)
            self._extension.deterministic_collective_stage(self._handle, input)
            self._synchronize_ranks()
            self._run_reduction(input, out, slice_offset=0)
            self._synchronize_ranks()
        return out

    def all_gather(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
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
            self._validate_matching_signature("all_gather", input)
            self._extension.deterministic_collective_stage(self._handle, input)
            self._synchronize_ranks()
            if self._backend == "cuda":
                self._extension.deterministic_collective_all_gather(self._handle, out)
            else:
                staged = self._staged_view(input)
                shard = input.size(0)
                slices = [out[index : index + shard] for index in range(0, out.size(0), shard)]
                dist.all_gather(slices, staged, group=self.group)
            self._synchronize_ranks()
        return out

    def reduce_scatter(
        self,
        input: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
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
            self._validate_matching_signature("reduce_scatter", input)
            self._extension.deterministic_collective_stage(self._handle, input)
            self._synchronize_ranks()
            if self._backend == "cuda":
                self._extension.deterministic_collective_reduce_scatter(self._handle, out)
            else:
                self._run_reduction(input, out, slice_offset=self.rank * out.numel())
            self._synchronize_ranks()
        return out

    # ------------------------------------------------------------------ #
    # Backend helpers
    # ------------------------------------------------------------------ #

    def _staged_view(self, input: torch.Tensor) -> torch.Tensor:
        """The staged input as a typed view of the staging buffer."""
        return self._staging.narrow(0, 0, input.numel() * input.element_size()).view(input.dtype)

    def _run_reduction(
        self,
        input: torch.Tensor,
        out: torch.Tensor,
        *,
        slice_offset: int,
    ) -> None:
        """NPU reduction: HCCL-gather every rank's staged input, then apply the
        fixed-tree kernel locally over the [world_size, N] gathered buffer."""
        gathered = torch.empty(
            (self.world_size, input.numel()),
            dtype=input.dtype,
            device=input.device,
        )
        dist.all_gather(
            list(gathered.unbind(0)),
            self._staged_view(input),
            group=self.group,
        )
        self._extension.deterministic_collective_reduce(
            self._handle,
            gathered,
            out,
            slice_offset,
        )

    def close(self) -> None:
        """Release the collective state (and CUDA IPC mappings) after the last call."""

        handle = getattr(self, "_handle", 0)
        if not handle:
            return
        if self._backend == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            torch.npu.synchronize(self.device)
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
        if input.device != self.device:
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
        if input.device != self.device:
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
        signatures: list[tuple[Any, ...] | None] = [None] * self.world_size
        dist.all_gather_object(signatures, signature, group=self.group)
        if any(peer_signature != signature for peer_signature in signatures):
            raise ValueError(
                f"all ranks must call {op_name} with matching shapes and dtypes; got {signatures}"
            )

    def _synchronize_ranks(self) -> None:
        if self._backend == "cuda":
            torch.cuda.synchronize(self.device)
            backend = dist.get_backend(self.group)
            if backend == dist.Backend.NCCL or str(backend).lower() == "nccl":
                dist.barrier(group=self.group, device_ids=[self.device.index])
            else:
                dist.barrier(group=self.group)
        else:
            torch.npu.synchronize(self.device)
            dist.barrier(group=self.group)
