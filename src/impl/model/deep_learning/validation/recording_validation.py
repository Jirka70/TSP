"""Extract and validate each recording before concatenating any epochs."""

from typing import Any

import numpy
from pydantic import ValidationError

from src.impl.model.deep_learning.validation.array_validation import (
    are_arrays_equal,
    are_numbers_close_to_each_other,
    convert_to_array,
)
from src.impl.model.deep_learning.validation.input_schemas import SignalSpecification, ValidatedRecording
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO


def has_epoch_container_interface(signal: object) -> bool:
    return callable(getattr(signal, "get_data", None))


def has_required_metadata(metadata: dict[str, Any]) -> bool:
    return all(name in metadata for name in ("signal_specification", "epoch_origins"))


def validate_dataset_recordings(dataset: EpochPreprocessedDTO, dataset_name: str) -> list[ValidatedRecording]:
    if not dataset.data:
        raise ValueError(f"{dataset_name}: dataset contains no recordings.")
    validated_recordings = []
    for index, recording in enumerate(dataset.data):
        validated_recordings.append(_validate_recording_with_location(recording, dataset_name, index))
    return validated_recordings


def _validate_recording_with_location(
    recording: RecordingDTO,
    dataset_name: str,
    recording_index: int,
) -> ValidatedRecording:
    try:
        return _extract_and_validate_recording(recording)
    except ValidationError as error:
        description = _describe_schema_validation_errors(error)
        raise ValueError(f"{dataset_name}, recording {recording_index}: {description}") from error
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError(f"{dataset_name}, recording {recording_index}: {error}") from error


def _describe_schema_validation_errors(error: ValidationError) -> str:
    descriptions = []
    for detail in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in detail["loc"]) or "recording"
        descriptions.append(f"{location}: {detail['msg']}")
    return "; ".join(descriptions)


def _extract_and_validate_recording(recording: RecordingDTO) -> ValidatedRecording:
    if not isinstance(recording.metadata, dict):
        raise ValueError("Recording metadata must be a dictionary.")
    if not has_required_metadata(recording.metadata):
        raise ValueError("Required metadata are missing: signal_specification and epoch_origins.")
    specification = SignalSpecification.model_validate(recording.metadata["signal_specification"])
    signal, labels = _extract_signal_and_labels(recording)
    _validate_available_signal_metadata(recording, specification)
    return ValidatedRecording(
        signal=signal,
        labels=labels,
        signal_specification=specification,
        epoch_origins=recording.metadata["epoch_origins"],
    )


def _extract_signal_and_labels(recording: RecordingDTO) -> tuple[numpy.ndarray, numpy.ndarray]:
    if has_epoch_container_interface(recording.data):
        return _extract_epoch_container_signal_and_labels(recording)
    if "labels" not in recording.metadata:
        raise ValueError("Recording labels are missing.")
    return convert_to_array(recording.data), convert_to_array(recording.metadata["labels"])


def _extract_epoch_container_signal_and_labels(recording: RecordingDTO) -> tuple[numpy.ndarray, numpy.ndarray]:
    signal = recording.data.get_data(copy=False)
    labels = recording.data.events[:, -1]
    if "labels" in recording.metadata:
        if not are_arrays_equal(labels, convert_to_array(recording.metadata["labels"])):
            raise ValueError("Event labels differ from metadata labels.")
    return convert_to_array(signal), convert_to_array(labels)


def _validate_available_signal_metadata(recording: RecordingDTO, specification: SignalSpecification) -> None:
    if has_epoch_container_interface(recording.data):
        _validate_epoch_container_description(recording.data, specification)
    if "ch_names" in recording.metadata:
        if tuple(recording.metadata["ch_names"]) != specification.channel_names:
            raise ValueError("Recording channel names or order differ from signal specification.")
    if "sfreq" in recording.metadata:
        if not are_numbers_close_to_each_other(float(recording.metadata["sfreq"]), specification.sampling_frequency):
            raise ValueError("Recording sampling frequency differs from signal specification.")


def _validate_epoch_container_description(epochs: Any, specification: SignalSpecification) -> None:
    if tuple(epochs.ch_names) != specification.channel_names:
        raise ValueError("Epoch container channel names or order differ from signal specification.")
    if not are_numbers_close_to_each_other(float(epochs.info["sfreq"]), specification.sampling_frequency):
        raise ValueError("Epoch container sampling frequency differs from signal specification.")
    if not are_numbers_close_to_each_other(float(epochs.tmin), specification.window_start_seconds):
        raise ValueError("Epoch container window start differs from signal specification.")
