"""Loading and consistency checking of the methodology rules catalog.

The catalog is stored in ``configs/methodology/`` and split into several
``rules_*.yaml`` files, one per methodology section (top-level key ``rules``).
All rule files are loaded in alphabetical order and merged into one
:class:`RulesCatalog`. See ``configs/methodology/_README.md`` for how to write rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter

from src.validation.methodology.check_registry import registered_checks
from src.validation.methodology.rule_model import MethodologyRule, RulesCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = PROJECT_ROOT / "configs" / "methodology"
RULE_FILES_PATTERN = "rules_*.yaml"

_RULES_ADAPTER = TypeAdapter(list[MethodologyRule])


class RulesCatalogError(ValueError):
    """Raised when the rules catalog is invalid."""


def load_rules_catalog(path: Path | str = DEFAULT_RULES_PATH) -> RulesCatalog:
    """Loads and validates the rules catalog.

    Args:
        path: Catalog directory (all ``rules_*.yaml`` files are merged), or a
            single YAML rule file.

    Besides the schema, it verifies that rule ids are unique across all files and
    that every ``check`` referenced by a rule is registered and belongs to the
    same phase as the rule.

    Raises:
        RulesCatalogError: a file is missing, malformed, or the catalog is inconsistent.
    """
    path = Path(path)
    if path.is_dir():
        rule_files = sorted(path.glob(RULE_FILES_PATTERN))
        if not rule_files:
            raise RulesCatalogError(f"No methodology rule files '{RULE_FILES_PATTERN}' found in '{path}'.")
    else:
        rule_files = [path]

    rules: list[MethodologyRule] = []
    for rule_file in rule_files:
        content = _read_mapping(rule_file, allowed_keys={"rules"})
        try:
            rules.extend(_RULES_ADAPTER.validate_python(content.get("rules") or []))
        except ValueError as e:
            raise RulesCatalogError(f"Invalid methodology rules '{rule_file}': {e}") from e

    try:
        catalog = RulesCatalog(rules=rules)
    except ValueError as e:
        raise RulesCatalogError(f"Invalid methodology rules in '{path}': {e}") from e

    return validate_catalog_checks(catalog)


def _read_mapping(path: Path, allowed_keys: set[str]) -> dict[str, Any]:
    """Reads a YAML mapping and rejects unexpected top-level keys."""
    try:
        with path.open(encoding="utf-8") as rules_file:
            raw = yaml.safe_load(rules_file)
    except (OSError, yaml.YAMLError) as e:
        raise RulesCatalogError(f"Cannot read methodology rules '{path}': {e}") from e

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RulesCatalogError(f"Methodology file '{path}' must contain a mapping, got {type(raw).__name__}.")
    unexpected = set(raw) - allowed_keys
    if unexpected:
        raise RulesCatalogError(f"Methodology file '{path}' has unexpected keys {sorted(unexpected)} (allowed: {sorted(allowed_keys)}).")
    return raw


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
