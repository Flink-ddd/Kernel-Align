# SPDX-License-Identifier: Apache-2.0  
# Copyright (c) 2026 RL-Engine Contributors

from typing import Optional, Tuple
from rl_engine.kernels.registry import kernel_registry
from rl_engine.utils.logger import logger

class RolloutExecutor:
    """
    Unified execution engine for RL rollout (sampling) phase.
    Decouples algorithmic logic from hardware-specific kernel implementations.
    """
    def __init__(self, model_config: Optional[dict] = None):
        self.config = model_config or {}
        logger.info("Initializing high-performance RolloutExecutor...")

    def _select_operators(self) -> Tuple[str, str]:
        """
        Automated hardware-aware operator selection.
        Ensures optimal throughput for large-batch GRPO rollout.
        """
        logp_be = kernel_registry.dispatch("fused_logp")
        attn_be = kernel_registry.dispatch("attention")
        
        logger.info_once(f"Selected Logp backend: {logp_be}")
        logger.info_once(f"Selected Attention backend: {attn_be}")
        
        return logp_be, attn_be

    def execute_rollout(self):
        """Execute the rollout process using selected optimized kernels."""
        logp_op, attn_op = self._select_operators()
        # Entry point for vLLM engine integration in future PRs
        logger.info_once(f"Starting rollout via {logp_op}...")
        return {"status": "success", "kernels": [logp_op, attn_op]}