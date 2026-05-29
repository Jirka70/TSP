from dataclasses import dataclass

from src.types.dto.evaluation.fold_evaluation_result_dto import FoldEvaluationResultDTO


@dataclass(frozen=True)
class EvaluationResultDTO:
    metrics: dict[str, float]
    fold_results: list[FoldEvaluationResultDTO] | None = None
    predictions: list[int] | None = None
    targets: list[int] | None = None
    probabilities: list[list[float]] | None = None
    confusion_matrix: list[list[int]] | None = None
    metadata: dict[str, object] | None = None
