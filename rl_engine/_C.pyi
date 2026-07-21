# rl_engine/_C.pyi
# This file is a type stub for the compiled C++ extension module.
import torch

def fused_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def fused_logp_sm90(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: ...
def fused_linear_logp_sm90(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None,
) -> list[torch.Tensor]: ...
def fused_linear_logp_sm90_global_target(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: torch.Tensor | None,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def fused_linear_logp_sm90_backward(
    grad_logp: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    lse: torch.Tensor,
    bias: torch.Tensor | None,
    vocab_start_index: int,
    compute_grad_hidden: bool,
    compute_grad_weight: bool,
    compute_grad_bias: bool,
    use_global_lse: bool,
) -> list[torch.Tensor]: ...
def linear_logp_probs_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_local_probs_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_local_bf16_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
) -> list[torch.Tensor]: ...
def linear_logp_probs_bf16_to_dlogits_(
    probs: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def linear_logp_local_probs_bf16_to_dlogits_(
    probs: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    local_lse: torch.Tensor,
    global_lse: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def linear_logp_logits_bf16_to_dlogits(
    logits: torch.Tensor,
    dlogits: torch.Tensor,
    target: torch.Tensor,
    grad_logp: torch.Tensor,
    lse: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor: ...
def fused_logp_forward_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_fp32(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def fused_logp_forward_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def fused_logp_forward_online_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor: ...
def deterministic_logp_forward_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_indexed_out(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor: ...
def deterministic_logp_forward_indexed_fp32(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor: ...
