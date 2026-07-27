// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#pragma once

// NVTX ranges are only meaningful -- and only safe to include -- on CUDA
// builds. csrc/ops.cpp also compiles under the ROCm/HIP build (its
// unconditional `fused_logp` binding has no #if guard), so this header must
// degrade to a true no-op there rather than failing to find <nvToolsExt.h>.
// ROCm/roctx tracing is explicit future work, not in scope here.
#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA) || defined(KERNEL_ALIGN_WITH_SM90)

#include <nvToolsExt.h>

#include <utility>

namespace rl_kernel {

// RAII scoped NVTX range. Uses the classic <nvToolsExt.h> API, which links
// against libnvToolsExt (see the `-lnvToolsExt` link flag added in setup.py)
// rather than nvtx3's dlopen-based injection layer. Calls are a cheap no-op
// when no profiler (nsys/ncu) is attached to the process.
class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

// Wraps a free-function pointer so pybind11 can bind the wrapper in place of
// the raw pointer; each call is bracketed by an NVTX range named `name`, so
// nsys shows one labeled block per RL-Kernel op regardless of how many CUDA
// kernels the op launches internally.
template <typename Ret, typename... Args>
auto traced(const char* name, Ret (*fn)(Args...)) {
  return [name, fn](Args... args) -> Ret {
    NvtxRange range(name);
    return fn(std::forward<Args>(args)...);
  };
}

}  // namespace rl_kernel

#define RL_KERNEL_NVTX_RANGE(name) ::rl_kernel::NvtxRange _rl_kernel_nvtx_range(name)

#else  // Not a CUDA build (e.g. ROCm-only): compile out entirely.

namespace rl_kernel {

template <typename Ret, typename... Args>
auto traced(const char* /*name*/, Ret (*fn)(Args...)) {
  return fn;
}

}  // namespace rl_kernel

#define RL_KERNEL_NVTX_RANGE(name) \
  do {                             \
  } while (0)

#endif
