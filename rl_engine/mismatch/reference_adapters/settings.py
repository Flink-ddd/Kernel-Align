# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Delivering pinned settings by channel, and reading them back.

Neither TransformerEngine nor FlashInfer is deterministic by default, so the
settings have to be pinned explicitly. But pinning alone is not enough: **a
setting that cannot be read back can only be recorded as ``UNOBSERVABLE``** --
delivered but unverifiable is the same as not delivered.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from rl_engine.mismatch.schema import RequiredSetting, SettingChannel, SwitchStatus


class SettingDeliveryError(RuntimeError):
    """A setting could not be delivered through its declared channel."""


def apply_required_settings(
    settings: Sequence[RequiredSetting],
    *,
    engine_kwargs: dict[str, Any] | None = None,
    call_kwargs: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> dict[SettingChannel, dict[str, Any]]:
    """Route each setting to the right place for its channel.

    Which channel a setting uses decides when it can take effect, and therefore
    its rebind cost -- an env var needs a process restart, a call argument does
    not. Flattening them into one dict leaves the framework unable to deliver
    them at all.
    """

    target_env = environ if environ is not None else os.environ
    delivered: dict[SettingChannel, dict[str, Any]] = {channel: {} for channel in SettingChannel}

    for setting in settings:
        if setting.channel is SettingChannel.ENV_VAR:
            target_env[setting.key] = str(setting.value)
        elif setting.channel is SettingChannel.TORCH_GLOBAL:
            _apply_torch_global(setting)
        elif setting.channel is SettingChannel.ENGINE_ARG:
            if engine_kwargs is None:
                raise SettingDeliveryError(
                    f"{setting.key!r} is an engine argument but no engine kwargs were given; "
                    "it can only take effect when the engine is rebuilt"
                )
            engine_kwargs[setting.key] = setting.value
        elif setting.channel is SettingChannel.CALL_ARG:
            if call_kwargs is None:
                raise SettingDeliveryError(
                    f"{setting.key!r} is a call argument but no call kwargs were given"
                )
            call_kwargs[setting.key] = setting.value
        delivered[setting.channel][setting.key] = setting.value

    return delivered


def _apply_torch_global(setting: RequiredSetting) -> None:
    """Set a ``torch.backends.*`` flag by dotted path."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        raise SettingDeliveryError(f"cannot set {setting.key!r} without torch") from exc

    target: Any = torch
    parts = setting.key.split(".")
    if parts[0] == "torch":
        parts = parts[1:]
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], setting.value)


def verify_required_settings(
    settings: Sequence[RequiredSetting],
    readback: Mapping[str, Any],
) -> tuple[SwitchStatus, tuple[str, ...]]:
    """Check the values that came back against what was pinned.

    Returns ``UNOBSERVABLE`` when a setting declares no readback path: it was
    delivered but cannot be proven, which is not evidence.
    """

    unobservable: list[str] = []
    mismatched: list[str] = []

    for setting in settings:
        if setting.readback is None:
            unobservable.append(setting.key)
            continue
        if setting.key not in readback:
            unobservable.append(setting.key)
            continue
        actual = readback[setting.key]
        if isinstance(setting.value, str) and setting.value.startswith(">="):
            continue  # a constraint rather than an exact value
        if actual != setting.value:
            mismatched.append(f"{setting.key}: pinned {setting.value!r}, read {actual!r}")

    if mismatched:
        return SwitchStatus.FELL_BACK, tuple(mismatched)
    if unobservable:
        return SwitchStatus.UNOBSERVABLE, tuple(f"{key}: no readback path" for key in unobservable)
    return SwitchStatus.APPLIED, ()


__all__ = [
    "SettingDeliveryError",
    "apply_required_settings",
    "verify_required_settings",
]
