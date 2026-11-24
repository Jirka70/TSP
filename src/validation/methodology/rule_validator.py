"""Validation entry point for the methodology rules catalog."""

from pathlib import Path

from src.validation.methodology.rule_loader import DEFAULT_RULES_PATH, load_rules_catalog
from src.validation.methodology.rule_model import RulesCatalog


class RuleValidator:
    """Loads methodology rule YAML files and validates them against the Pydantic schemas."""

    def __init__(self, path: Path | str = DEFAULT_RULES_PATH) -> None:
        self._path = path

    def validate(self) -> RulesCatalog:
        """Return the validated catalog or raise ``RulesCatalogError`` for invalid rules."""
        return load_rules_catalog(self._path)
