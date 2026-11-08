"""Built-in methodology checks. Importing this package registers them."""

from src.validation.methodology.checks import runtime_checks, static_checks  # noqa: F401

__all__ = ["runtime_checks", "static_checks"]
