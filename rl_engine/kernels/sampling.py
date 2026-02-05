# SPDX-License-Identifier: Apache-2.0  
# Copyright (c) 2026 RL-Engine Contributors

import torch
import torch.nn as nn
from rl_engine.utils.logger import logger
from rl_engine.platforms.constants import constants

class SamplerBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.backend = self._detect_backend()
        self._init_backend_assets()

    def _detect_backend(self):
        if torch.version.hip:
            logger.info_once("Detected AMD GPU (ROCm) - Using AITER backend")
            return constants.BackendLib.AITER.value
        else:
            logger.info_once("Detected NVIDIA GPU (CUDA) - Using FlashInfer backend")
            return constants.BackendLib.FLASHINFER.value

    def _init_backend_assets(self):
        """
        Preload backend-specific dependencies 
        to avoid runtime import overhead.
        """
        if self.backend == constants.BackendLib.FLASHINFER.value:
            try:
                import flashinfer
                self.flashinfer = flashinfer
                logger.info("FlashInfer kernels loaded successfully.")
            except ImportError:
                logger.error("FlashInfer not found. Please install it for NVIDIA GPUs.")
        elif self.backend == constants.BackendLib.AITER.value:
            pass

    @torch.inference_mode()
    def sample(self, logits, top_k=None, top_p=None, temperature=1.0, deterministic=True):
        """
        Unified sampling interface.
        """
        logits = logits.contiguous()
        if temperature != 1.0:
            logits = logits / temperature

        if self.backend == constants.BackendLib.FLASHINFER.value:
            from flashinfer.sampling import (
                sampling_from_logits, 
                top_k_top_p_sampling_from_logits,
                top_p_sampling_from_probs,
                top_k_renorm_probs
            )

            logits = logits.float().contiguous()
            if temperature != 1.0:
                logits.div_(temperature)
            if top_k is None and top_p is None:
                return sampling_from_logits(logits, deterministic=deterministic)
            
            if top_k is not None and top_p is not None:
                return top_k_top_p_sampling_from_logits(
                    logits, top_k, top_p, deterministic=deterministic
                )
            
            if top_p is not None:
                probs = torch.softmax(logits, dim=-1)
                return top_p_sampling_from_probs(probs, top_p, deterministic=deterministic)
            
            if top_k is not None:
                probs = torch.softmax(logits, dim=-1)
                renorm_probs = top_k_renorm_probs(probs, top_k)
                return torch.multinomial(renorm_probs, num_samples=1).view(-1)
            
            return sampling_from_logits(logits, deterministic=deterministic)

            
            
        elif self.backend == constants.BackendLib.AITER.value:
            # TODO: Connect to AITER's sampling operator
            # return aiter.ops.sample(logits, ...)
            pass

        # Fallback to native PyTorch sampling
        if top_k is not None:
            topk_values, _ = torch.topk(logits, top_k)
            min_topk = topk_values[..., -1, None]
            logits = torch.where(logits < min_topk, torch.full_like(logits, float('-inf')), logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def compute_logp(self, logits, token_ids):
        """
        Unified Logprob interface.
        Core advantage: avoids generating intermediate [Batch, Seq, Vocab] float32 matrices.
        """
        logits = logits.contiguous()

        if self.backend == constants.BackendLib.FLASHINFER.value:
            from flashinfer.sampling import softmax
            probs = softmax(logits, temperature=1.0)
            logp = torch.log(probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1))
            return logp
            
        elif self.backend == constants.BackendLib.AITER.value:
            try:
                import aiter
                # return aiter.ops.logp(logits, token_ids)
                raise NotImplementedError("AITER logp operator integration in progress...")
            except ImportError:
                return torch.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

        return torch.log_softmax(logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)