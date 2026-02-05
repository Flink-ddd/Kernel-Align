# SPDX-License-Identifier: Apache-2.0  
# Copyright (c) 2026 RL-Engine Contributors

from rl_engine.utils.logger import logger
from rl_engine.platforms.device import device_ctx
from rl_engine.kernels.registry import kernel_registry
from rl_engine.executors.rollout import RolloutExecutor

def test_logger_enhancements():
    logger.info("Testing standard info log.")
    
    print("Next message should only appear ONCE even with 3 calls:")
    for i in range(3):
        logger.info_once("This is a unique message that should appear only once.")

def test_device_and_registry():
    logger.info(f"Detected Device: {device_ctx.device_type} (ROCm: {device_ctx.is_rocm})")
    
    logp_op = kernel_registry.dispatch("fused_logp")
    attn_op = kernel_registry.dispatch("attention")
    logger.info(f"Dispatched Logp Operator: {logp_op}")
    logger.info(f"Dispatched Attention Operator: {attn_op}")

def test_executor_flow():
    executor = RolloutExecutor()
    result = executor.execute_rollout()
    logger.info(f"Executor result: {result}")

if __name__ == "__main__":
    try:
        test_logger_enhancements()
        test_device_and_registry()
        test_executor_flow()
        print("\n All infrastructure tests passed!")
    except Exception as e:
        print(f"\n Test failed with error: {e}")