# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Python-side NVTX stage ranges.

Stage ranges (``rlk::rollout.generate``, ``rlk::train_step``, ...) group the
per-operator ``rlk::<op>`` ranges emitted by the C++ bindings on an Nsight
Systems timeline. Uses ``torch.cuda.nvtx`` (no extra dependency) and silently
no-ops when CUDA is unavailable or a profiler API is missing.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache
from typing import Any, Optional

RL_KERNEL_NVTX = "RL_KERNEL_NVTX"


@lru_cache(maxsize=1)
def _nvtx_enabled() -> bool:
    """Return whether runtime NVTX emission was explicitly requested."""
    return os.environ.get(RL_KERNEL_NVTX, "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _nvtx_module() -> Optional[Any]:
    if not _nvtx_enabled():
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        nvtx = torch.cuda.nvtx
        if not hasattr(nvtx, "range_push") or not hasattr(nvtx, "range_pop"):
            return None
        return nvtx
    except Exception:
        return None


class _NvtxRange(AbstractContextManager[None]):
    def __init__(self, nvtx: Any, name: str) -> None:
        self._nvtx = nvtx
        self._name = name

    def __enter__(self) -> None:
        self._nvtx.range_push(self._name)

    def __exit__(self, *_exc_info: Any) -> None:
        self._nvtx.range_pop()


def nvtx_range(name: str) -> AbstractContextManager[None]:
    """Return an opted-in NVTX range, or a cheap null context by default."""
    nvtx = _nvtx_module()
    if nvtx is None:
        return nullcontext()
    return _NvtxRange(nvtx, name)
