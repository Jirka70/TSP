"""Loading and consistency checking of the methodology rules catalog."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.validation.methodology.check_registry import registered_checks
from src.validation.methodology.rule_model import RulesCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = PROJECT_ROOT / "configs" / "methodology" / "rules.yaml"


class RulesCatalogError(ValueError):
    """Raised when the rules catalog is invalid."""


def load_rules_catalog(path: Path | str = DEFAULT_RULES_PATH) -> RulesCatalog:
    """Loads and validates the rules catalog.

    Besides the schema, it verifies that every ``check`` referenced by a rule is
    registered and belongs to the same phase as the rule.

    Raises:
        RulesCatalogError: the file is missing, malformed, or inconsistent.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as rules_file:
            raw = yaml.safe_load(rules_file)
    except OSError as e:
        raise RulesCatalogError(f"Cannot read methodology rules '{path}': {e}") from e

    try:
        catalog = RulesCatalog.model_validate(raw)
    except ValueError as e:
        raise RulesCatalogError(f"Invalid methodology rules '{path}': {e}") from e

    return validate_catalog_checks(catalog)


def validate_catalog_checks(catalog: RulesCatalog) -> RulesCatalog:
    """Verifies that referenced Python checks exist and match the rule phase."""
    checks = registered_checks()
    errors: list[str] = []
    for rule in catalog.rules:
        if rule.check is None:
            continue
        registered = checks.get(rule.check)
        if registered is None:
            errors.append(f"rule '{rule.id}' references unknown check '{rule.check}'")
        elif registered.phase != rule.phase:
            errors.append(f"rule '{rule.id}' is '{rule.phase.value}' but check '{rule.check}' is '{registered.phase.value}'")

    if errors:
        raise RulesCatalogError("Inconsistent methodology rules: " + "; ".join(errors))
    return catalog
