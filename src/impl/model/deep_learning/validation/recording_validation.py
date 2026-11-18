"""Extract and validate each recording before concatenating any epochs."""

import numpy
from numpy.typing import NDArray
from pydantic import ValidationError

from src.impl.model.deep_learning.validation.array_validation import (
    are_arrays_equal,
    convert_to_array,
)
from src.impl.model.deep_learning.validation.input_schemas import ValidatedRecording
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO


def has_epoch_container_interface(signal: object) -> bool:
    return callable(getattr(signal, "get_data", None))


def validate_dataset_recordings(dataset: EpochPreprocessedDTO, dataset_name: str) -> list[ValidatedRecording]:
    if not dataset.data:
        raise ValueError(f"{dataset_name}: dataset contains no recordings.")
    validated_recordings = []
    for recording_index, recording in enumerate(dataset.data):
        validated_recordings.append(_validate_recording_with_location(recording, dataset_name, recording_index))
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
    signal, labels = _extract_signal_and_labels(recording)
    return ValidatedRecording(signal=signal, labels=labels)


def _extract_signal_and_labels(recording: RecordingDTO) -> tuple[NDArray[numpy.generic], NDArray[numpy.generic]]:
    if has_epoch_container_interface(recording.data):
        return _extract_epoch_container_signal_and_labels(recording)
    if "labels" not in recording.metadata:
        raise ValueError("Recording labels are missing.")
    return convert_to_array(recording.data), convert_to_array(recording.metadata["labels"])


def _extract_epoch_container_signal_and_labels(
    recording: RecordingDTO,
) -> tuple[NDArray[numpy.generic], NDArray[numpy.generic]]:
    signal = recording.data.get_data(copy=False)
    labels = recording.data.events[:, -1]
    if "labels" not in recording.metadata:
        return convert_to_array(signal), convert_to_array(labels)
    if not are_arrays_equal(labels, convert_to_array(recording.metadata["labels"])):
        raise ValueError("Event labels differ from metadata labels.")
    return convert_to_array(signal), convert_to_array(labels)
