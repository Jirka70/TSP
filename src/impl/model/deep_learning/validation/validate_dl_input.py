"""Validate training and validation arrays before model construction or training."""

from collections import Counter

from src.impl.model.deep_learning.validation.array_validation import concatenate_recording_arrays
from src.impl.model.deep_learning.validation.input_schemas import ValidatedDeepLearningInput, ValidatedRecording
from src.impl.model.deep_learning.validation.recording_validation import validate_dataset_recordings
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.learning_dataset import LearningDataset

RecordingsByDataset = dict[str, list[ValidatedRecording]]


def validate_dl_input(input_data: FinalTrainingInputDTO) -> ValidatedDeepLearningInput:
    if not isinstance(input_data.config, EEGNetConfig):
        raise ValueError("FinalEEGNetTrainer requires EEGNetConfig.")

    datasets = {"train": input_data.training_data, "val": input_data.validation_data}
    recordings: RecordingsByDataset = {}
    for dataset_name, dataset in datasets.items():
        if dataset is None:
            raise ValueError(f"{dataset_name}: dataset is missing.")
        recordings[dataset_name] = validate_dataset_recordings(dataset, dataset_name)

    reference_shape = recordings["train"][0].signal.shape[1:]
    class_counts: dict[str, dict[int, int]] = {}
    for dataset_name, dataset_recordings in recordings.items():
        _validate_matching_channel_and_sample_counts(dataset_name, dataset_recordings, reference_shape)
        class_counts[dataset_name] = _count_dataset_classes(dataset_recordings)

    if len(class_counts["train"]) != input_data.config.n_classes:
        raise ValueError("Training class count differs from configured class count.")

    warnings = _validate_validation_classes_and_report_missing_classes(class_counts)
    return ValidatedDeepLearningInput(
        training=_combine_validated_recordings(recordings["train"]),
        validation=_combine_validated_recordings(recordings["val"]),
        class_counts=class_counts,
        warnings=warnings,
    )


def _validate_matching_channel_and_sample_counts(
    dataset_name: str,
    recordings: list[ValidatedRecording],
    reference_shape: tuple[int, ...],
) -> None:
    for recording_index, recording in enumerate(recordings):
        if recording.signal.shape[1:] != reference_shape:
            raise ValueError(
                f"{dataset_name}, recording {recording_index}: channel or time sample count differs from train."
            )


def _count_dataset_classes(recordings: list[ValidatedRecording]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for recording in recordings:
        counts.update(recording.labels.tolist())
    return dict(sorted(counts.items()))


def _validate_validation_classes_and_report_missing_classes(class_counts: dict[str, dict[int, int]]) -> list[str]:
    training_classes = set(class_counts["train"])
    validation_classes = set(class_counts["val"])
    unknown_classes = validation_classes - training_classes
    if unknown_classes:
        raise ValueError(f"Validation contains classes absent from train: {sorted(unknown_classes)}.")

    missing_classes = training_classes - validation_classes
    if missing_classes:
        return [f"val: no examples for classes {sorted(missing_classes)}."]
    return []


def _combine_validated_recordings(recordings: list[ValidatedRecording]) -> LearningDataset:
    return LearningDataset(
        x=concatenate_recording_arrays([recording.signal for recording in recordings]),
        y=concatenate_recording_arrays([recording.labels for recording in recordings]),
    )
