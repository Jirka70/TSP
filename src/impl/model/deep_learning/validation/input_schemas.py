"""Validated signal descriptions, original signal provenance and recording arrays."""

from typing import Literal

import numpy
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator
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


class MetadataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def has_unique_values(values: tuple[object, ...]) -> bool:
    return len(set(values)) == len(values)


def has_only_nonempty_identifiers(identifiers: tuple[str, ...]) -> bool:
    return all(bool(identifier.strip()) for identifier in identifiers)


class SignalSpecification(MetadataSchema):
    class_identifiers: tuple[StrictInt, ...] = Field(min_length=2)
    label_encoding_identifier: StrictStr = Field(min_length=1)
    channel_names: tuple[StrictStr, ...] = Field(min_length=1)
    sampling_frequency: float = Field(gt=0, strict=True)
    unit: StrictStr = Field(min_length=1)
    representation: StrictStr = Field(min_length=1)
    number_of_time_samples: StrictInt = Field(gt=0)
    window_start_seconds: float = Field(strict=True)
    preprocessing_identifier: StrictStr = Field(min_length=1)
    split_protocol: Literal["trial", "session", "subject"]
    preprocessing_fit_trial_identifiers: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        if not has_unique_values(self.class_identifiers):
            raise ValueError("Class identifiers contain duplicates.")
        if not has_unique_values(self.channel_names):
            raise ValueError("Channel names contain duplicates.")
        if not has_only_nonempty_identifiers(self.channel_names):
            raise ValueError("Channel names must not be blank.")
        if not has_only_nonempty_identifiers(self.preprocessing_fit_trial_identifiers):
            raise ValueError("Preprocessing fit trial identifiers must not be blank.")
        return self


class OriginalSignalInterval(MetadataSchema):
    """Globally scoped identifiers and [start, stop) in the ORIGINAL source sample coordinates."""

    subject_identifier: StrictStr = Field(min_length=1)
    session_identifier: StrictStr = Field(min_length=1)
    source_identifier: StrictStr = Field(min_length=1)
    trial_identifier: StrictStr = Field(min_length=1)
    start_sample: StrictInt = Field(ge=0)
    stop_sample: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def validate_sample_interval(self) -> Self:
        if self.stop_sample <= self.start_sample:
            raise ValueError("Original signal interval must have positive length.")
        return self


class EpochOrigin(MetadataSchema):
    roots: tuple[OriginalSignalInterval, ...] = Field(min_length=1)
    augmented: StrictBool


class ValidatedRecording(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    signal: numpy.ndarray
    labels: numpy.ndarray
    signal_specification: SignalSpecification
    epoch_origins: tuple[EpochOrigin, ...]

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
    if len(recording.signal) != len(recording.epoch_origins):
        raise ValueError("Epoch and origin counts differ.")
    if recording.signal.shape[1] != len(recording.signal_specification.channel_names):
        raise ValueError("Channel count differs from signal specification.")
    if recording.signal.shape[2] != recording.signal_specification.number_of_time_samples:
        raise ValueError("Time sample count differs from signal specification.")


def _validate_recording_values(recording: ValidatedRecording) -> None:
    if not has_only_real_numbers(recording.signal):
        raise ValueError("Signal must contain real numbers.")
    if not has_only_finite_numbers(recording.signal):
        raise ValueError("Signal contains NaN or infinity.")
    if not has_only_integer_numbers(recording.labels):
        raise ValueError("Labels must be integers.")
    unknown_classes = set(recording.labels.tolist()) - set(recording.signal_specification.class_identifiers)
    if unknown_classes:
        raise ValueError(f"Undeclared class identifiers: {sorted(unknown_classes)}.")


def _copy_signal_at_training_precision(signal: numpy.ndarray) -> numpy.ndarray:
    converted_signal = copy_as_training_precision(signal)
    if not has_only_finite_numbers(converted_signal):
        raise ValueError("Signal contains NaN or infinity after float32 conversion.")
    return converted_signal


class ValidatedDeepLearningInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    training: LearningDataset
    validation: LearningDataset
    test: LearningDataset
    signal_specification: SignalSpecification
    class_counts: dict[str, dict[int, int]]
    warnings: list[str]
