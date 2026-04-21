"""Configuration DTOs for data splitting strategies."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.types.dto.config.astageconfig import AStageConfig


class SplitConfig(AStageConfig):
    """
    Configuration for the basic percentage-based splitting strategy.

    Attributes:
        backend: Literal identifier for the 'basic' splitter.
        enabled: Whether this splitting strategy is enabled.
        train_ratio: Fraction of data used for training (0.0 - 1.0).
        validation_ratio: Fraction of data used for validation (0.0 - 1.0).
        test_ratio: Fraction of data used for testing (0.0 - 1.0).
        shuffle: Whether to shuffle data before splitting.
        random_seed: Seed for the random number generator to ensure reproducibility.
    """

    backend: Literal["basic"]
    enabled: bool

    train_ratio: float
    validation_ratio: float
    test_ratio: float

    shuffle: bool
    random_seed: int


class MoabbWithinSessionSplit(BaseModel):
    """
    Evaluator settings for MOABB WithinSessionSplitter.

    Attributes:
        target: The target MOABB class path (aliased to '_target_').
        random_state: Seed for reproducibility.
        n_folds: Number of folds for cross-validation.
        shuffle: Whether to shuffle before splitting (depends on cv_class).
        cv_class: The scikit-learn cross-validation class (e.g., StratifiedKFold).
    """

    target: str = Field(alias="_target_")
    random_state: int | None = None
    n_folds: int | None = None
    shuffle: bool = True
    cv_class: Any | None = None


class MoabbWithinSubjectSplit(BaseModel):
    """
    Evaluator settings for MOABB WithinSubjectSplitter.

    Attributes:
        target: The target MOABB class path (aliased to '_target_').
        random_state: Seed for reproducibility.
        n_folds: Number of folds for cross-validation.
        shuffle: Whether to shuffle before splitting (depends on cv_class).
        cv_class: The scikit-learn cross-validation class (e.g., StratifiedKFold).
    """

    target: str = Field(alias="_target_")
    random_state: int | None = None
    n_folds: int | None = None
    shuffle: bool = True
    cv_class: Any | None = None


class MoabbCrossSubjectSplit(BaseModel):
    """
    Evaluator settings for MOABB CrossSubjectSplitter.

    Attributes:
        target: The target MOABB class path (aliased to '_target_').
        random_state: Seed for reproducibility.
        cv_class: The scikit-learn cross-validation class (e.g., LeaveOneGroupOut).
    """

    target: str = Field(alias="_target_")
    random_state: int | None = None
    cv_class: Any | None = None


class MoabbCrossSessionSplit(BaseModel):
    """
    Evaluator settings for MOABB CrossSessionSplitter.

    Attributes:
        target: The target MOABB class path (aliased to '_target_').
        random_state: Seed for reproducibility.
        cv_class: The scikit-learn cross-validation class (e.g., LeaveOneGroupOut).
        shuffle: Whether to shuffle before splitting.
    """

    target: str = Field(alias="_target_")
    random_state: int | None = None
    cv_class: Any | None = None
    shuffle: bool = True


class SplitMoabbWithinSessionConfig(AStageConfig):
    """
    Full stage configuration for MOABB Within-Session splitting.
    """

    backend: Literal["moabb_within_session"]
    enabled: bool
    evaluator: MoabbWithinSessionSplit


class SplitMoabbWithinSubjectConfig(AStageConfig):
    """
    Full stage configuration for MOABB Within-Subject splitting.
    """

    backend: Literal["moabb_within_subject"]
    enabled: bool
    evaluator: MoabbWithinSubjectSplit


class SplitMoabbCrossSubjectConfig(AStageConfig):
    """
    Full stage configuration for MOABB Cross-Subject splitting.
    """

    backend: Literal["moabb_cross_subject"]
    enabled: bool
    evaluator: MoabbCrossSubjectSplit


class SplitMoabbCrossSessionConfig(AStageConfig):
    """
    Full stage configuration for MOABB Cross-Session splitting.
    """

    backend: Literal["moabb_cross_session"]
    enabled: bool
    evaluator: MoabbCrossSessionSplit
