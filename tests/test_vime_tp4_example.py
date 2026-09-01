# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from examples.vime_qwen3_8b_tp4_cp2_200.run_arm import (
    MEGATRON_ATTENTION_BACKEND,
)


def test_tp4_formal_matrix_pins_the_vime_qwen3_attention_backend():
    assert MEGATRON_ATTENTION_BACKEND == "fused"
