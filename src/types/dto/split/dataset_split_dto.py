from __future__ import annotations

from dataclasses import dataclass, field

from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class FoldDTO:
    """
    Represents a single cross-validation fold or data partition.

    Used by the evaluation pipeline for cross-validation.

    Attributes:
        fold_idx: Index of the fold (starting from 0).
        train_data: Preprocessed data to be used for model training.
        test_data: Preprocessed data to be used for final model evaluation.
    """

    fold_idx: int
    train_data: EpochPreprocessedDTO
    test_data: EpochPreprocessedDTO | None


@dataclass(frozen=True)
class DatasetSplitDTO:  # TODO: necheceme toto DTO prejmenovat na AugmentedDataDTO / AugmentedDatasetSplitDTO
    """
    Main output of the splitting stage.

    Supports two modes depending on the pipeline:

    - **Training pipeline** (basic splitter): ``folds`` is ``None``;
      ``train``, ``test`` and ``validation`` carry the data partitions directly.
    - **Evaluation pipeline** (MOABB / cross-validation splitter): ``train``,
      ``test`` and ``validation`` are ``None``; ``folds`` holds the CV folds.

    Attributes:
        folds: Cross-validation folds produced by the evaluation pipeline splitter.
            ``None`` when the training pipeline is used.
        train: Training data for the final model (training pipeline only).
            ``None`` when the evaluation pipeline is used.
        test: Held-out test data for final model evaluation (training pipeline only).
            May be ``None`` if ``test_ratio`` is 0.0.
        validation: Validation data used during DL epoch training (training pipeline only).
            May be ``None`` if ``validation_ratio`` is 0.0 (e.g. for ML models).
    """

    folds: list[FoldDTO] | None = field(default=None)
    train: EpochPreprocessedDTO | None = field(default=None)
    test: EpochPreprocessedDTO | None = field(default=None)
    validation: EpochPreprocessedDTO | None = field(default=None)
