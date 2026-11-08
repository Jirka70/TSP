"""Evaluation of declarative rule conditions over a dumped experiment config."""

from __future__ import annotations

from typing import Any

from src.validation.methodology.rule_model import Condition


class _Missing:
    """Sentinel for a path that does not exist in the config."""

    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()


def resolve_path(data: Any, path: str) -> Any:
    """Resolves a dot-separated path in nested dicts / lists.

    Integer segments index into lists. Returns ``MISSING`` when any segment
    does not exist.
    """
    current = data
    for segment in path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return MISSING
            current = current[segment]
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return MISSING
        else:
            return MISSING
    return current


def evaluate_condition(condition: Condition, data: dict[str, Any]) -> bool:
    """Evaluates a condition against the dumped config (``model_dump(mode="json")``)."""
    if condition.all_ is not None:
        return all(evaluate_condition(child, data) for child in condition.all_)
    if condition.any_ is not None:
        return any(evaluate_condition(child, data) for child in condition.any_)
    if condition.not_ is not None:
        return not evaluate_condition(condition.not_, data)

    assert condition.path is not None  # guaranteed by Condition.validate_shape
    value = resolve_path(data, condition.path)
    operator = condition.operator

    if operator == "exists":
        return (value is not MISSING) == condition.exists
    if value is MISSING:
        return False

    if operator == "eq":
        return value == condition.eq
    if operator == "ne":
        return value != condition.ne
    if operator == "in_":
        return value in (condition.in_ or [])
    if operator == "not_in":
        return value not in (condition.not_in or [])
    if operator == "contains":
        return isinstance(value, (list, tuple, str)) and condition.contains in value
    if operator in ("gt", "ge", "lt", "le"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        threshold = getattr(condition, operator)
        return {
            "gt": value > threshold,
            "ge": value >= threshold,
            "lt": value < threshold,
            "le": value <= threshold,
        }[operator]

    raise ValueError(f"Unsupported condition operator: {operator}")
