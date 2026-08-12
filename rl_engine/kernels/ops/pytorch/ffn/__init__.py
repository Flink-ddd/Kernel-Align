# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Composable Qwen-style feed-forward network reference operators."""

from .tensor_parallel import FFNContext, TensorParallelFFN, shard_qwen3_ffn_weights

__all__ = ["FFNContext", "TensorParallelFFN", "shard_qwen3_ffn_weights"]
