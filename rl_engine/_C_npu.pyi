# rl_engine/_C_npu.pyi
# Type stub for the compiled Ascend C (CANN) extension module.
# Built only when KERNEL_ALIGN_FORCE_ASCEND=1 on a machine with CANN + torch_npu.
import torch

def batch_invariant_logp_ascend(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
) -> list[torch.Tensor]: ...
def deterministic_collective_create(
    staging: torch.Tensor,
    world_size: int,
    rank: int,
) -> int: ...
def deterministic_collective_destroy(handle: int) -> None: ...
def deterministic_collective_stage(handle: int, input: torch.Tensor) -> None: ...
def deterministic_collective_reduce(
    handle: int,
    gathered: torch.Tensor,
    output: torch.Tensor,
    slice_offset: int,
) -> None: ...
