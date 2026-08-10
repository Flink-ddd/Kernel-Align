# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reference implementations shared by attention's factors."""

from __future__ import annotations

from rl_engine.mismatch.schema import (
    ExecutionPath,
    LibraryPin,
    ReferenceAuthority,
    ReferenceImplementation,
    RequiredSetting,
    SettingChannel,
)

TE_ROPE_REFERENCE = ReferenceImplementation(
    name="transformer_engine",
    tier=ReferenceAuthority.SHARED_BACKEND,
    training_impl="transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb",
    rollout_impl="flashinfer.rope.apply_rope",
    covers_paths=(
        ExecutionPath.TRAINING_FULL_PREFILL,
        ExecutionPath.ROLLOUT_FULL_PREFILL,
    ),
    required_settings=(
        RequiredSetting(
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            "0",
            SettingChannel.ENV_VAR,
            readback="os.environ",
        ),
    ),
    pinned_libraries=(LibraryPin("transformer_engine", "2.9.0.dev0", commit="8260f49"),),
)
