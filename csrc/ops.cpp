// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>
#include <cuda_bf16.h>

// Fused LogP Declarations
torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids);

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_SM90)
torch::Tensor fused_logp_sm90_forward(torch::Tensor logits, torch::Tensor labels);
std::vector<torch::Tensor> fused_linear_logp_sm90_forward(torch::Tensor hidden,
                                                          torch::Tensor weight,
                                                          torch::Tensor target,
                                                          torch::optional<torch::Tensor> bias);
std::vector<torch::Tensor> fused_linear_logp_sm90_global_target_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::optional<torch::Tensor> bias,
    int64_t vocab_start_index);
std::vector<torch::Tensor> fused_linear_logp_sm90_backward(torch::Tensor grad_logp,
                                                           torch::Tensor hidden,
                                                           torch::Tensor weight,
                                                           torch::Tensor target,
                                                           torch::Tensor lse,
                                                           torch::optional<torch::Tensor> bias,
                                                           int64_t vocab_start_index,
                                                           bool compute_grad_hidden,
                                                           bool compute_grad_weight,
                                                           bool compute_grad_bias,
                                                           bool use_global_lse);
std::vector<torch::Tensor> linear_logp_probs_bf16_forward(torch::Tensor logits,
                                                          torch::Tensor target,
                                                          int64_t vocab_start_index);
std::vector<torch::Tensor> linear_logp_bf16_forward(torch::Tensor logits,
                                                    torch::Tensor target,
                                                    int64_t vocab_start_index);
std::vector<torch::Tensor> linear_logp_local_probs_bf16_forward(torch::Tensor logits,
                                                                torch::Tensor target,
                                                                int64_t vocab_start_index);
std::vector<torch::Tensor> linear_logp_local_bf16_forward(torch::Tensor logits,
                                                          torch::Tensor target,
                                                          int64_t vocab_start_index);
torch::Tensor linear_logp_probs_bf16_to_dlogits_(torch::Tensor probs,
                                                 torch::Tensor target,
                                                 torch::Tensor grad_logp,
                                                 int64_t vocab_start_index);
torch::Tensor linear_logp_local_probs_bf16_to_dlogits_(torch::Tensor probs,
                                                       torch::Tensor target,
                                                       torch::Tensor grad_logp,
                                                       torch::Tensor local_lse,
                                                       torch::Tensor global_lse,
                                                       int64_t vocab_start_index);
torch::Tensor linear_logp_logits_bf16_to_dlogits(torch::Tensor logits,
                                                 torch::Tensor dlogits,
                                                 torch::Tensor target,
                                                 torch::Tensor grad_logp,
                                                 torch::Tensor lse,
                                                 int64_t vocab_start_index);
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA)
torch::Tensor fused_logp_forward_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor fused_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_indexed_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices, torch::Tensor output);
torch::Tensor fused_logp_forward_indexed_fp32(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);
torch::Tensor fused_logp_forward_online_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor fused_logp_forward_online_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_online_indexed_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices, torch::Tensor output);
torch::Tensor fused_logp_forward_online_indexed_fp32(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);
torch::Tensor deterministic_logp_forward(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor deterministic_logp_forward_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor deterministic_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor deterministic_logp_forward_indexed_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices, torch::Tensor output);
torch::Tensor deterministic_logp_forward_indexed_fp32(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);

// Prefix-Shared Attention Declarations & Wrappers

void prefix_shared_attention_forward(
  const __nv_bfloat16 *Q,  // [bs, G, len_q, DIM]
  const __nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const __nv_bfloat16 *V,  // [bs, len_kv, DIM]
  __nv_bfloat16 *O,        // [bs, G, len_q, DIM]
  int bs,
  int G,
  int len_q,
  int len_kv,
  int dim);

at::Tensor prefix_shared_attention(
  const at::Tensor& Q,
  const at::Tensor& K,
  const at::Tensor& V)
{
  TORCH_CHECK(Q.dim() == 4, "Q must be [bs, G, len_q, DIM]");
  TORCH_CHECK(K.dim() == 3, "K must be [bs, len_kv, DIM]");
  TORCH_CHECK(V.dim() == 3, "V must be [bs, len_kv, DIM]");

  TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Only BFloat16 is supported");
  TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Tensors must be CUDA and contiguous");
  TORCH_CHECK(K.is_cuda() && K.is_contiguous(), "Tensors must be CUDA and contiguous");
  TORCH_CHECK(V.is_cuda() && V.is_contiguous(), "Tensors must be CUDA and contiguous");

  const int bs = Q.size(0);
  const int G = Q.size(1);
  const int len_q = Q.size(2);
  const int dim = Q.size(3);
  const int len_kv = K.size(1);

  at::Tensor O = at::empty_like(Q);

  auto Q_ptr = reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr());
  auto K_ptr = reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr());
  auto V_ptr = reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr());
  auto O_ptr = reinterpret_cast<__nv_bfloat16 *>(O.data_ptr());

  prefix_shared_attention_forward(Q_ptr, K_ptr, V_ptr, O_ptr, bs, G, len_q, len_kv, dim);

  return O;
}
#endif

// PyBind11 Module Registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RL-Kernel High-Performance Operator Extension Library";

    m.def("fused_logp", &fused_logp_forward, "Fused logp forward fallback");

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_SM90)
    m.def("fused_logp_sm90", &fused_logp_sm90_forward, "TMA-accelerated Online Softmax Fused LogP");
    m.def("fused_linear_logp_sm90", &fused_linear_logp_sm90_forward,
          "TMA+WGMMA fused linear log-prob (hidden @ W^T -> selected-token logp), SM90");
    m.def("fused_linear_logp_sm90_global_target", &fused_linear_logp_sm90_global_target_forward,
          "TMA+WGMMA local-shard target-logit/lse for vocab-parallel linear log-prob, SM90");
    m.def("fused_linear_logp_sm90_backward", &fused_linear_logp_sm90_backward,
          "CUDA fused backward for linear log-prob, SM90 backend");
    m.def("linear_logp_probs_bf16_forward", &linear_logp_probs_bf16_forward,
          "Build bf16 softmax probabilities and selected log-prob from bf16 logits");
    m.def("linear_logp_bf16_forward", &linear_logp_bf16_forward,
          "Build selected log-prob and lse from bf16 logits without saving probabilities");
    m.def("linear_logp_local_probs_bf16_forward", &linear_logp_local_probs_bf16_forward,
          "Build local bf16 softmax probabilities, target logits, and lse from bf16 logits");
    m.def("linear_logp_local_bf16_forward", &linear_logp_local_bf16_forward,
          "Build local target logits and lse from bf16 logits without saving probabilities");
    m.def("linear_logp_probs_bf16_to_dlogits_", &linear_logp_probs_bf16_to_dlogits_,
          "In-place bf16 probs -> dlogits for selected log-prob backward");
    m.def("linear_logp_local_probs_bf16_to_dlogits_",
          &linear_logp_local_probs_bf16_to_dlogits_,
          "In-place local bf16 probs -> TP dlogits for selected log-prob backward");
    m.def("linear_logp_logits_bf16_to_dlogits", &linear_logp_logits_bf16_to_dlogits,
          "Build bf16 dlogits from bf16 logits and fp32 lse");
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA)
    m.def("fused_logp_forward_out", &fused_logp_forward_out, "Fused logp out");
    m.def("fused_logp_forward_fp32", &fused_logp_forward_fp32, "Fused logp fp32");
    m.def("fused_logp_forward_indexed_out", &fused_logp_forward_indexed_out, "Fused logp indexed out");
    m.def("fused_logp_forward_indexed_fp32", &fused_logp_forward_indexed_fp32, "Fused logp indexed fp32");
    m.def("fused_logp_forward_online_out", &fused_logp_forward_online_out, "Fused logp online out");
    m.def("fused_logp_forward_online_fp32", &fused_logp_forward_online_fp32, "Fused logp online fp32");
    m.def("fused_logp_forward_online_indexed_out", &fused_logp_forward_online_indexed_out, "Fused logp online indexed out");
    m.def("fused_logp_forward_online_indexed_fp32", &fused_logp_forward_online_indexed_fp32, "Fused logp online indexed fp32");
    m.def("deterministic_logp", &deterministic_logp_forward, "Batch-invariant deterministic logp");
    m.def("deterministic_logp_forward_out", &deterministic_logp_forward_out, "Batch-invariant deterministic logp out");
    m.def("deterministic_logp_forward_fp32", &deterministic_logp_forward_fp32, "Batch-invariant deterministic logp fp32");
    m.def("deterministic_logp_forward_indexed_out", &deterministic_logp_forward_indexed_out, "Batch-invariant deterministic logp indexed out");
    m.def("deterministic_logp_forward_indexed_fp32", &deterministic_logp_forward_indexed_fp32, "Batch-invariant deterministic logp indexed fp32");

    // registry Prefix-Shared Attention
    m.def("prefix_shared_attention", &prefix_shared_attention, "Prefix-Shared Fused Attention for GRPO");
#endif
}
