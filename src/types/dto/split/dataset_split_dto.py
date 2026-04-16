from __future__ import annotations

from dataclasses import dataclass

from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class FoldDTO:
    """
    Represents a single cross-validation fold or data partition.

    Attributes:
        fold_idx: Index of the fold (starting from 0).
        train_data: Preprocessed data to be used for model training.
        validation_data: Preprocessed data to be used for hyperparameter tuning (optional).
        test_data: Preprocessed data to be used for final model evaluation.
    """

    fold_idx: int
    train_data: EpochPreprocessedDTO
    validation_data: EpochPreprocessedDTO | None
    test_data: EpochPreprocessedDTO | None


@dataclass(frozen=True)
class DatasetSplitDTO:
    """
    Main output of the splitting stage, containing all generated folds.

    Attributes:
        folds: A list of FoldDTO objects representing the partitions of the dataset.
    """

    folds: list[FoldDTO]
