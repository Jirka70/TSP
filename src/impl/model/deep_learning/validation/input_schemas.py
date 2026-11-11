"""Validated training arrays without requirements on signal provenance metadata."""

import numpy
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from src.impl.model.deep_learning.validation.array_validation import (
    copy_as_training_precision,
    has_only_finite_numbers,
    has_only_integer_numbers,
    has_only_positive_dimensions,
    has_only_real_numbers,
    is_array_with_dimensions,
)
from src.types.dto.model.learning_dataset import LearningDataset


class ValidatedRecording(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    signal: NDArray[numpy.generic]
    labels: NDArray[numpy.generic]

    @model_validator(mode="after")
    def validate_arrays_and_copy_signal(self) -> Self:
        _validate_recording_dimensions(self)
        _validate_recording_values(self)
        self.signal = _copy_signal_at_training_precision(self.signal)
        return self


def _validate_recording_dimensions(recording: ValidatedRecording) -> None:
    if not is_array_with_dimensions(recording.signal, 3):
        raise ValueError("Signal must have shape [epochs, channels, samples].")
    if not has_only_positive_dimensions(recording.signal):
        raise ValueError("Signal dimensions must be positive.")
    if not is_array_with_dimensions(recording.labels, 1):
        raise ValueError("Labels must have shape [epochs].")
    if len(recording.signal) != len(recording.labels):
        raise ValueError("Epoch and label counts differ.")


def _validate_recording_values(recording: ValidatedRecording) -> None:
    if not has_only_real_numbers(recording.signal):
        raise ValueError("Signal must contain real numbers.")
    if not has_only_finite_numbers(recording.signal):
        raise ValueError("Signal contains NaN or infinity.")
    if not has_only_integer_numbers(recording.labels):
        raise ValueError("Labels must be integers.")


def _copy_signal_at_training_precision(signal: NDArray[numpy.generic]) -> NDArray[numpy.generic]:
    converted_signal = copy_as_training_precision(signal)
    if not has_only_finite_numbers(converted_signal):
        raise ValueError("Signal contains NaN or infinity after float32 conversion.")
    return converted_signal


class ValidatedDeepLearningInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    training: LearningDataset
    validation: LearningDataset
    class_counts: dict[str, dict[int, int]]
    warnings: list[str]
