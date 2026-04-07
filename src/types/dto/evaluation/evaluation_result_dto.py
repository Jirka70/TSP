from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResultDTO:
    metrics: dict[str, float]
    predictions: list[int] | None = None
    targets: list[int] | None = None
    probabilities: list[list[float]] | None = None
    confusion_matrix: list[list[int]] | None = None
    metadata: dict[str, object] | None = None
