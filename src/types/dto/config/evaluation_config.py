from typing import Literal

from pydantic import field_validator
from sklearn.metrics import get_scorer_names

from src.types.dto.config.astageconfig import AStageConfig


class EvaluationConfig(AStageConfig):
    backend: Literal["default"]
    metrics: list[str]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validates that all metrics are supported by scikit-learn."""
        valid_names = get_scorer_names()
        for metric in v:
            if metric not in valid_names:
                raise ValueError(f"Metric '{metric}' is not a valid scikit-learn scorer name. Valid options are: {valid_names}")
        return v


class SklearnEvaluationConfig(AStageConfig):
    backend: Literal["sklearn"]
    metrics: list[str]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[str]) -> list[str]:
        """Validates that all metrics are supported by scikit-learn."""
        valid_names = get_scorer_names()
        for metric in v:
            if metric not in valid_names:
                raise ValueError(f"Metric '{metric}' is not a valid scikit-learn scorer name. Valid options are: {valid_names}")
        return v
