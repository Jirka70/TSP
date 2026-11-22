"""Built-in methodology checks. Importing this package registers them."""

from src.validation.methodology.checks import (  # noqa: F401
    preprocessing_checks,
    runtime_checks,
    static_checks,
    training_run_checks,
)

__all__ = ["preprocessing_checks", "runtime_checks", "static_checks", "training_run_checks"]
