// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxBlocks = 65535;

__device__ __forceinline__ float swiglu_fp32(float gate, float up) {
  const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  return (gate * sigmoid_gate) * up;
}

__global__ void swiglu_forward_kernel(const __nv_bfloat16 *__restrict__ gate,
                                      const __nv_bfloat16 *__restrict__ up,
                                      __nv_bfloat16 *__restrict__ output,
                                      int64_t numel) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x * 2;
  for (int64_t index =
           (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
       index < numel; index += stride) {
    if (index + 1 < numel) {
      const auto gate2 =
          reinterpret_cast<const __nv_bfloat162 *>(gate)[index / 2];
      const auto up2 = reinterpret_cast<const __nv_bfloat162 *>(up)[index / 2];
      reinterpret_cast<__nv_bfloat162 *>(output)[index / 2] =
          __floats2bfloat162_rn(
              swiglu_fp32(__low2float(gate2), __low2float(up2)),
              swiglu_fp32(__high2float(gate2), __high2float(up2)));
    } else {
      output[index] = __float2bfloat16_rn(swiglu_fp32(
          __bfloat162float(gate[index]), __bfloat162float(up[index])));
    }
  }
}

void check_bf16_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, name,
              " must have dtype torch.bfloat16");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) %
                      alignof(__nv_bfloat162) ==
                  0,
              name, " must be 4-byte aligned for bfloat162 access");
}

void check_sm90(const torch::Tensor &tensor) {
  c10::cuda::CUDAGuard device_guard(tensor.device());
  const auto *properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 9,
              "SwiGLU SM90 kernel requires Hopper compute capability 9.x, got "
              "sm_",
              properties->major, properties->minor);
}

int launch_blocks(int64_t numel) {
  const int64_t work_items = (numel + 1) / 2;
  return static_cast<int>(
      std::min<int64_t>((work_items + kThreads - 1) / kThreads, kMaxBlocks));
}

} // namespace

torch::Tensor swiglu_forward_sm90(torch::Tensor gate, torch::Tensor up) {
  check_bf16_tensor(gate, "gate");
  check_bf16_tensor(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes(),
              "gate and up must have the same shape");
  TORCH_CHECK(gate.device() == up.device(),
              "gate and up must be on the same device");
  check_sm90(gate);

  auto output = torch::empty_like(gate);
  if (gate.numel() == 0) {
    return output;
  }

  c10::cuda::CUDAGuard device_guard(gate.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  swiglu_forward_kernel<<<launch_blocks(gate.numel()), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(gate.data_ptr()),
      reinterpret_cast<const __nv_bfloat16 *>(up.data_ptr()),
      reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()), gate.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
