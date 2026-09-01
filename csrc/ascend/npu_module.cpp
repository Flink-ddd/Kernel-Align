// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Pybind entry point for the rl_engine._C_npu extension. The Ascend C kernels
// and their torch host wrappers live in the sibling *.asc files; this TU only
// declares and binds them so every Ascend op shares one compiled module.
//
// Build: see setup.py (AscendBuildExtension, bisheng -x asc), gated by
// KERNEL_ALIGN_FORCE_ASCEND=1. Requires CANN toolkit + torch_npu.

#include <torch/extension.h>

std::vector<torch::Tensor> batch_invariant_logp_ascend_forward(torch::Tensor logits,
                                                               torch::Tensor target,
                                                               int64_t ignore_index);

torch::Tensor lm_head_ascend_forward(torch::Tensor hidden,
                                     torch::Tensor weight,
                                     torch::optional<torch::Tensor> bias,
                                     bool output_fp32);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batch_invariant_logp_ascend",
          &batch_invariant_logp_ascend_forward,
          "Batch-invariant selected-token log-probability (Ascend C forward)");
    m.def("lm_head_ascend",
          &lm_head_ascend_forward,
          "Batch-invariant LM-head projection (Ascend C forward)");
}
