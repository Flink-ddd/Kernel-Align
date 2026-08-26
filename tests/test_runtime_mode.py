import pytest

from rl_engine.runtime_mode import (
    RLKernelMode,
    rl_kernel_mode,
    rl_kernel_mode_policy,
)


@pytest.mark.parametrize(
    ("mode", "enabled", "case_id", "provider_mode", "fail_on_fallback"),
    (
        ("strict", True, "R/R", "strict", True),
        ("audit", True, "R/R", "strict", False),
        ("auto", True, "P/P", "auto", False),
        ("off", False, "P/P", None, False),
    ),
)
def test_mode_policy(mode, enabled, case_id, provider_mode, fail_on_fallback):
    policy = rl_kernel_mode_policy(mode)

    assert policy.mode is RLKernelMode(mode)
    assert policy.enabled is enabled
    assert policy.case_id == case_id
    assert policy.provider_mode == provider_mode
    assert policy.fail_on_fallback is fail_on_fallback


def test_mode_rejects_unknown_value():
    with pytest.raises(RuntimeError, match="RL_KERNEL_MODE must be one of"):
        rl_kernel_mode("unexpected")
