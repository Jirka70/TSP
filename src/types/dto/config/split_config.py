"""Configuration DTOs for data splitting strategies."""

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

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
        exclude_validation_data_before_split: Whether to extract validation data before the main split (based on subjects or samples).
    """

    backend: Literal["basic"]
    enabled: bool

    train_ratio: float = Field(ge=0.0, le=1.0)
    validation_ratio: float = Field(gt=0.0, le=1.0)
    test_ratio: float = Field(ge=0.0, le=1.0)

    shuffle: bool
    random_seed: int = Field(ge=0)
    exclude_validation_data_before_split: bool

    @model_validator(mode="after")
    def validate_ratios_sum(self) -> "SplitBasicConfig":
        """Ensures that the sum of split ratios is approximately 1.0."""
        if self.enabled:
            total = self.train_ratio + self.validation_ratio + self.test_ratio
            epsilon = 0.01
            if abs(1.0 - total) > epsilon:
                raise ValueError(f"The sum of train, validation, and test ratios must be 1.0 (current sum: {total})")
        return self


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
    random_state: int | None = Field(default=None, ge=0)
    n_folds: int = Field(ge=2)
    shuffle: bool = True
    cv_class: Any | None = None

    @model_validator(mode="after")
    def validate_random_state_without_shuffle(self) -> "MoabbWithinSessionSplit":
        """Ensures random_state is None if shuffle is False."""
        if not self.shuffle and self.random_state is not None:
            raise ValueError("random_state must be Null if shuffle is False (scikit-learn requirement).")
        return self


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
    random_state: int | None = Field(default=None, ge=0)
    n_folds: int = Field(ge=2)
    shuffle: bool = True
    cv_class: Any | None = None

    @model_validator(mode="after")
    def validate_random_state_without_shuffle(self) -> "MoabbWithinSubjectSplit":
        """Ensures random_state is None if shuffle is False."""
        if not self.shuffle and self.random_state is not None:
            raise ValueError("random_state must be Null if shuffle is False (scikit-learn requirement).")
        return self


class MoabbCrossSubjectSplit(BaseModel):
    """
    Evaluator settings for MOABB CrossSubjectSplitter.

    Attributes:
        target: The target MOABB class path (aliased to '_target_').
        random_state: Seed for reproducibility.
        cv_class: The scikit-learn cross-validation class (e.g., LeaveOneGroupOut).
    """

    target: str = Field(alias="_target_")
    random_state: int = Field(ge=0)
    cv_class: Any


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
    random_state: int | None = Field(default=None, ge=0)
    cv_class: Any | None = None
    shuffle: bool = True

    @model_validator(mode="after")
    def validate_shuffle_cv_class(self) -> "MoabbCrossSessionSplit":
        """Ensures shuffle=False when using LeaveOneGroupOut and handles random_state."""
        cv = str(self.cv_class) if self.cv_class else ""
        # Default in MOABB for CrossSession is LeaveOneGroupOut if cv_class is None
        is_leave_one_group_out = not self.cv_class or "LeaveOneGroupOut" in cv

        if self.shuffle and is_leave_one_group_out:
            raise ValueError("Shuffling is not supported for LeaveOneGroupOut in CrossSessionSplitter. Set shuffle=False or use a different cv_class (e.g., GroupShuffleSplit).")

        if not self.shuffle and self.random_state is not None:
            raise ValueError("random_state must be Null if shuffle is False (scikit-learn requirement).")

        return self


class MoabbSplitConfig(AStageConfig):
    """
    Base configuration for MOABB-based splitting strategies.

    Attributes:
        enabled: Whether this splitting strategy is enabled.
        exclude_validation_data_before_split: Whether to extract validation data before MOABB splitting (based on subjects).
        validation_ratio: Fraction of data (subjects or samples) to use for validation.
    """

    enabled: bool
    exclude_validation_data_before_split: bool = False
    validation_ratio: float = Field(default=0.0, gt=0.0, le=1.0)


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
