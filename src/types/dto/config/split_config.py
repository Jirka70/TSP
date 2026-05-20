"""Configuration DTOs for data splitting strategies."""

from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from src.types.dto.config.astageconfig import AStageConfig


class SplitBasicConfig(AStageConfig):
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
        pre_split_validation: Whether to extract validation data before the main split (based on subjects or samples).
    """

    backend: Literal["basic"]
    enabled: bool

    train_ratio: float
    validation_ratio: float
    test_ratio: float

    shuffle: bool
    random_seed: int
    pre_split_validation: bool


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


class MoabbSplitConfig(AStageConfig):
    """
    Base configuration for MOABB-based splitting strategies.

    Attributes:
        enabled: Whether this splitting strategy is enabled.
        pre_split_validation: Whether to extract validation data before MOABB splitting (based on subjects).
        validation_ratio: Fraction of data (subjects or samples) to use for validation.
    """

    enabled: bool
    pre_split_validation: bool = False
    validation_ratio: float = 0.0


class SplitMoabbWithinSessionConfig(MoabbSplitConfig):
    """
    Full stage configuration for MOABB Within-Session splitting.
    """

    backend: Literal["moabb_within_session"]
    evaluator: MoabbWithinSessionSplit


class SplitMoabbWithinSubjectConfig(MoabbSplitConfig):
    """
    Full stage configuration for MOABB Within-Subject splitting.
    """

    backend: Literal["moabb_within_subject"]
    evaluator: MoabbWithinSubjectSplit


class SplitMoabbCrossSubjectConfig(MoabbSplitConfig):
    """
    Full stage configuration for MOABB Cross-Subject splitting.
    """

    backend: Literal["moabb_cross_subject"]
    evaluator: MoabbCrossSubjectSplit


class SplitMoabbCrossSessionConfig(MoabbSplitConfig):
    """
    Full stage configuration for MOABB Cross-Session splitting.
    """

    backend: Literal["moabb_cross_session"]
    evaluator: MoabbCrossSessionSplit


SplitConfig = Union[
    SplitBasicConfig,
    SplitMoabbWithinSessionConfig,
    SplitMoabbWithinSubjectConfig,
    SplitMoabbCrossSubjectConfig,
    SplitMoabbCrossSessionConfig,
]
