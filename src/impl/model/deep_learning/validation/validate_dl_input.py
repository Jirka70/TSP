"""Validate all three datasets before model construction or training."""

from collections import Counter

from src.impl.model.deep_learning.validation.array_validation import concatenate_recording_arrays
from src.impl.model.deep_learning.validation.dataset_separation import (
    OriginalIntervalsByDataset,
    validate_original_intervals_do_not_overlap,
    validate_original_signal_identities,
    validate_preprocessing_was_fitted_only_on_training_trials,
    validate_protocol_separation,
)
from src.impl.model.deep_learning.validation.input_schemas import (
    OriginalSignalInterval,
    SignalSpecification,
    ValidatedDeepLearningInput,
    ValidatedRecording,
)
from src.impl.model.deep_learning.validation.recording_validation import validate_dataset_recordings
from src.types.dto.model.dl_input_dto import DLInputDTO
from src.types.dto.model.learning_dataset import LearningDataset


RecordingsByDataset = dict[str, list[ValidatedRecording]]


def validate_dl_input(input_data: DLInputDTO) -> ValidatedDeepLearningInput:
    recordings: RecordingsByDataset = {}
    for dataset_name in ("train", "val", "test"):
        dataset = getattr(input_data, dataset_name)
        if dataset is None:
            raise ValueError(f"{dataset_name}: dataset is missing.")
        recordings[dataset_name] = validate_dataset_recordings(dataset, dataset_name)

    specification = recordings["train"][0].signal_specification
    if len(specification.class_identifiers) != input_data.config.n_classes:
        raise ValueError("Declared class count differs from configured class count.")

    class_counts: dict[str, dict[int, int]] = {}
    original_intervals: OriginalIntervalsByDataset = {}
    for dataset_name, dataset_recordings in recordings.items():
        _require_matching_recording_specifications(dataset_name, dataset_recordings, specification)
        class_counts[dataset_name] = _count_dataset_classes(dataset_recordings, specification)
        original_intervals[dataset_name] = _collect_dataset_original_intervals(dataset_recordings)
        if dataset_name == "train":
            continue
        if has_augmented_epochs(dataset_recordings):
            raise ValueError(f"{dataset_name}: evaluation dataset contains training augmentation.")

    warnings = _validate_training_classes_and_report_missing_evaluation_classes(class_counts)
    validate_original_signal_identities(original_intervals)
    validate_protocol_separation(original_intervals, specification)
    validate_original_intervals_do_not_overlap(original_intervals)
    validate_preprocessing_was_fitted_only_on_training_trials(original_intervals, specification)

    return ValidatedDeepLearningInput(
        training=_combine_validated_recordings(recordings["train"]),
        validation=_combine_validated_recordings(recordings["val"]),
        test=_combine_validated_recordings(recordings["test"]),
        signal_specification=specification,
        class_counts=class_counts,
        warnings=warnings,
    )


def _require_matching_recording_specifications(
    dataset_name: str,
    recordings: list[ValidatedRecording],
    reference: SignalSpecification,
) -> None:
    for index, recording in enumerate(recordings):
        if recording.signal_specification != reference:
            raise ValueError(
                f"{dataset_name}, recording {index}: signal specification differs from training reference."
            )


def _count_dataset_classes(
    recordings: list[ValidatedRecording],
    specification: SignalSpecification,
) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for recording in recordings:
        counts.update(recording.labels.tolist())
    return {identifier: counts.get(identifier, 0) for identifier in specification.class_identifiers}


def _validate_training_classes_and_report_missing_evaluation_classes(
    class_counts: dict[str, dict[int, int]],
) -> list[str]:
    missing_training_classes = _get_missing_classes(class_counts["train"])
    if missing_training_classes:
        raise ValueError(f"Train lacks declared classes: {missing_training_classes}.")
    warnings = []
    for dataset_name, counts in class_counts.items():
        if has_missing_classes(counts):
            warnings.append(f"{dataset_name}: no examples for classes {_get_missing_classes(counts)}.")
    return warnings


def has_missing_classes(class_counts: dict[int, int]) -> bool:
    return any(count == 0 for count in class_counts.values())


def _get_missing_classes(class_counts: dict[int, int]) -> list[int]:
    return sorted(identifier for identifier, count in class_counts.items() if count == 0)


def has_augmented_epochs(recordings: list[ValidatedRecording]) -> bool:
    return any(origin.augmented for recording in recordings for origin in recording.epoch_origins)


def _collect_dataset_original_intervals(recordings: list[ValidatedRecording]) -> list[OriginalSignalInterval]:
    intervals = []
    for recording in recordings:
        for origin in recording.epoch_origins:
            intervals.extend(origin.roots)
    return intervals


def _combine_validated_recordings(recordings: list[ValidatedRecording]) -> LearningDataset:
    return LearningDataset(
        x=concatenate_recording_arrays([recording.signal for recording in recordings]),
        y=concatenate_recording_arrays([recording.labels for recording in recordings]),
    )
