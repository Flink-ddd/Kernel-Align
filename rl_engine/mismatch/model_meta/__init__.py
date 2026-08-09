# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Model shape and the module correspondence table.

Filled once per model by whoever brings it up, and shared by every operator:
"Megatron's ``linear_fc1`` corresponds to vLLM's ``gate_up_proj``" is the same
fact for attention, gemm and logprob alike. Swapping in GLM5 or DSv3.2 means
redoing it -- which is why it lives here and not in any ``operator_checks/``.
"""

from rl_engine.mismatch.model_meta.qwen3 import QWEN3_CORRESPONDENCES, QWEN3_EDGES
