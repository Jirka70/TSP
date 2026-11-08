"""Registry of Python methodology checks referenced from the rules catalog via ``check: <name>``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.types.dto.config.experiment_config import ExperimentConfig
from src.validation.methodology.rule_model import RulePhase


@dataclass(frozen=True)
class Finding:
    """One concrete occurrence of a rule violation.

    Attributes:
        detail: What exactly was found (human-readable).
        location: Config path (or data location) the finding relates to.
    """

    detail: str
    location: str | None = None


@dataclass(frozen=True)
class CheckContext:
    """Input of a methodology check.

    Attributes:
        config: Validated experiment configuration.
        params: Rule parameters from the catalog (``params:``).
        data: Runtime data for runtime checks (e.g. ``DatasetSplitDTO``); ``None`` for static checks.
    """

    config: ExperimentConfig
    params: dict[str, Any] = field(default_factory=dict)
    data: Any = None


CheckFn = Callable[[CheckContext], list[Finding]]


@dataclass(frozen=True)
class RegisteredCheck:
    """A check function together with the phase it belongs to."""

    name: str
    phase: RulePhase
    fn: CheckFn


_REGISTRY: dict[str, RegisteredCheck] = {}


def methodology_check(name: str, phase: RulePhase = RulePhase.STATIC) -> Callable[[CheckFn], CheckFn]:
    """Registers a function as a methodology check under ``name``.

    The function receives a ``CheckContext`` and returns a list of findings
    (an empty list means the rule is satisfied).
    """

    def decorator(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY:
            raise ValueError(f"Methodology check '{name}' is already registered.")
        _REGISTRY[name] = RegisteredCheck(name=name, phase=phase, fn=fn)
        return fn

    return decorator


def get_check(name: str) -> RegisteredCheck:
    """Returns a registered check or raises ``KeyError``."""
    _ensure_builtin_checks_loaded()
    return _REGISTRY[name]


def registered_checks() -> dict[str, RegisteredCheck]:
    """Returns a copy of all registered checks."""
    _ensure_builtin_checks_loaded()
    return dict(_REGISTRY)


def _ensure_builtin_checks_loaded() -> None:
    # Importing the package registers the built-in checks via the decorator.
    import src.validation.methodology.checks  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
