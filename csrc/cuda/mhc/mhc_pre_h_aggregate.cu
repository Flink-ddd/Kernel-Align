#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "mhc_pre_h_aggregate_kernel.cuh"

torch::Tensor mhc_pre_h_aggregate_cuda(torch::Tensor residual,
                                       torch::Tensor pre) {
  TORCH_CHECK(residual.is_cuda() && pre.is_cuda(),
              "residual and pre must be CUDA tensors");
  TORCH_CHECK(residual.device() == pre.device(),
              "residual and pre must be on the same CUDA device");
  TORCH_CHECK(residual.is_contiguous() && pre.is_contiguous(),
              "residual and pre must be contiguous");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16,
              "residual must be bfloat16");
  TORCH_CHECK(pre.scalar_type() == torch::kFloat32,
              "pre must be float32");
  TORCH_CHECK(residual.dim() == 3 && residual.size(1) == 4,
              "residual must have shape [num_tokens, 4, hidden_size]");
  TORCH_CHECK(pre.dim() == 2 && pre.size(1) == 4,
              "pre must have shape [num_tokens, 4]");
  TORCH_CHECK(residual.size(0) == pre.size(0),
              "residual and pre must have the same num_tokens");

  int64_t const num_tokens = residual.size(0);
  int64_t const hidden_size = residual.size(2);
  TORCH_CHECK(num_tokens <= std::numeric_limits<unsigned int>::max(),
              "num_tokens exceeds the CUDA grid limit");

  auto output = torch::empty({num_tokens, hidden_size}, residual.options());
  if (num_tokens == 0 || hidden_size == 0) {
    return output;
  }

  c10::cuda::CUDAGuard const device_guard(residual.device());
  int device = 0;
  int major = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device));
  TORCH_CHECK(major >= 8, "mhc_pre_h_aggregate requires SM80 or newer");

  auto const* residual_ptr = reinterpret_cast<__nv_bfloat16 const*>(
      residual.data_ptr<at::BFloat16>());
  auto const* pre_ptr = pre.data_ptr<float>();
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(
      output.data_ptr<at::BFloat16>());

  cudaError_t const status = rl_kernel::mhc::launch_mhc_pre_h_aggregate(
      residual_ptr, pre_ptr, output_ptr, num_tokens, hidden_size,
      at::cuda::getCurrentCUDAStream(), major >= 9);
  C10_CUDA_CHECK(status);
  return output;
}
