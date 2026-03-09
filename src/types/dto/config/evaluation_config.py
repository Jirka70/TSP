from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: list[str]