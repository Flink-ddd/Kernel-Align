# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plugin registration and factor discovery.

Adding an operator is adding a directory; adding a factor is adding a file. If
you find yourself editing the planner, a global dict, or another operator's file,
the abstraction is missing something -- raise it rather than patching in place.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Mapping, Protocol

from rl_engine.mismatch.schema import (
    CollectiveContract,
    ComparisonRule,
    ImplementationResolution,
    MismatchFactor,
    OperatorContract,
    PolicyRole,
)


class OperatorChecks(Protocol):
    """Everything one operator needs checked. The plugin itself.

    These four are operator-level, not factor-level: reading configuration back
    from an engine is the same logic for all of an operator's factors. Factor
    files hold declarations only.
    """

    operator: str

    def declare_factors(self) -> tuple[MismatchFactor, ...]:
        """Which factors this operator has."""
        ...

    def build_contract(
        self, role: PolicyRole, switch_values: Mapping[str, Any]
    ) -> OperatorContract:
        """Turn this side's switch values into that side's numerical contract."""
        ...

    def read_effective_config(self, role: PolicyRole, adapter: Any) -> Mapping[str, Any]:
        """Read the switches' effective values back. A requested value is not
        evidence."""
        ...

    def observe_collectives(self, role: PolicyRole, adapter: Any) -> tuple[CollectiveContract, ...]:
        """Which collectives actually ran."""
        ...

    def resolve_implementation(
        self, factor_id: str, role: PolicyRole, impl_name: str
    ) -> tuple[Callable[..., Any] | None, ImplementationResolution]:
        """Resolve the name a variant asks for into a callable.

        Return the trace even when resolution fails: a bare ``None`` leaves a
        silent fallback with nothing to investigate.
        """
        ...


class RegistrationError(ValueError):
    """A plugin conflicts with one already registered."""


class PluginRegistry:
    """Registered operator plugins, with conflict detection at import time."""

    def __init__(self) -> None:
        self._plugins: dict[str, OperatorChecks] = {}

    def register(self, plugin_cls: type) -> type:
        """Instantiate and register a plugin, checking for conflicts."""

        plugin = plugin_cls()
        name = getattr(plugin, "operator", "")
        if not name:
            raise RegistrationError(f"{plugin_cls.__name__} does not declare an operator name")
        if name in self._plugins:
            raise RegistrationError(f"operator {name!r} is already registered")

        self._check_factor_conflicts(plugin)
        self._plugins[name] = plugin
        return plugin_cls

    def _check_factor_conflicts(self, plugin: OperatorChecks) -> None:
        """Reject duplicate ids, duplicate switch paths, and one contract field
        claimed at two different comparison rules."""

        known_ids = {factor.id for p in self._plugins.values() for factor in p.declare_factors()}
        known_paths = {
            factor.switch.path for p in self._plugins.values() for factor in p.declare_factors()
        }
        rules: dict[str, ComparisonRule] = {}
        for existing in self._plugins.values():
            for factor in existing.declare_factors():
                rules.update(factor.comparison_rules)

        local_ids: set[str] = set()
        local_paths: set[str] = set()
        for factor in plugin.declare_factors():
            if factor.id in known_ids:
                raise RegistrationError(f"duplicate factor id {factor.id!r}")
            if factor.switch.path in known_paths:
                raise RegistrationError(f"duplicate switch path {factor.switch.path!r}")
            if factor.id in local_ids:
                raise RegistrationError(f"duplicate factor id {factor.id!r}")
            if factor.switch.path in local_paths:
                raise RegistrationError(f"duplicate switch path {factor.switch.path!r}")
            for field_path, rule in factor.comparison_rules.items():
                previous = rules.get(field_path)
                if previous is not None and previous is not rule:
                    raise RegistrationError(
                        f"contract field {field_path!r} is declared as {previous.value!r} "
                        f"elsewhere but {rule.value!r} by {factor.id!r}"
                    )
                rules[field_path] = rule
            local_ids.add(factor.id)
            local_paths.add(factor.switch.path)

    def operators(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def plugin(self, operator: str) -> OperatorChecks:
        if operator not in self._plugins:
            raise KeyError(f"operator {operator!r} is not registered; known: {self.operators()}")
        return self._plugins[operator]

    def factors_for(self, operator: str | None = None) -> tuple[MismatchFactor, ...]:
        """All factors, or just one operator's."""

        if operator is not None:
            return tuple(self.plugin(operator).declare_factors())
        collected: list[MismatchFactor] = []
        for name in self.operators():
            collected.extend(self._plugins[name].declare_factors())
        return tuple(collected)

    def clear(self) -> None:
        """Test helper: drop every registration."""

        self._plugins.clear()


OPERATOR_CHECKS = PluginRegistry()


class FactorDiscoveryError(ValueError):
    """A factor module does not follow the discovery convention."""


def discover_factors(package: str) -> tuple[MismatchFactor, ...]:
    """Collect the ``FACTOR`` constant from every module under ``<package>.factors``.

    A file's name must equal its factor id with the operator prefix stripped, so
    that renaming an id without renaming the file fails at import rather than
    silently dropping the factor.
    """

    factors_package = f"{package}.factors"
    module = importlib.import_module(factors_package)
    collected: list[MismatchFactor] = []

    for info in pkgutil.iter_modules(module.__path__):
        if info.name.startswith("_"):
            continue
        submodule = importlib.import_module(f"{factors_package}.{info.name}")
        factor = getattr(submodule, "FACTOR", None)
        if factor is None:
            raise FactorDiscoveryError(f"{factors_package}.{info.name} defines no FACTOR constant")
        expected_suffix = factor.id.split(".", 1)[-1]
        if info.name != expected_suffix:
            raise FactorDiscoveryError(
                f"{factors_package}.{info.name}.py declares factor id {factor.id!r}; "
                f"the file should be named {expected_suffix}.py"
            )
        collected.append(factor)

    return tuple(sorted(collected, key=lambda item: item.id))


__all__ = [
    "FactorDiscoveryError",
    "OPERATOR_CHECKS",
    "OperatorChecks",
    "PluginRegistry",
    "RegistrationError",
    "discover_factors",
]
