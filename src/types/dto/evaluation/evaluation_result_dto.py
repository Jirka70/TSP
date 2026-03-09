from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResultDTO:
    metrics: dict[str, float]
    predictions: object
    metadata: dict[str, object]