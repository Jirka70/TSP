"""Small disjoint datasets with the complete deep-learning input contract."""

import numpy

from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO


def create_signal_specification():
    return {
        "class_identifiers": [0, 1],
        "label_encoding_identifier": "synthetic-classes-v1",
        "channel_names": ["C3", "C4"],
        "sampling_frequency": 32.0,
        "unit": "V",
        "representation": "eeg",
        "number_of_time_samples": 32,
        "window_start_seconds": 0.0,
        "preprocessing_identifier": "fixed-filter-v1",
        "split_protocol": "trial",
        "preprocessing_fit_trial_identifiers": [],
    }


def create_epoch_origin(dataset_identifier, epoch_index):
    return {
        "augmented": False,
        "roots": [
            {
                "subject_identifier": "synthetic/person-1",
                "session_identifier": "synthetic/person-1/session-1",
                "source_identifier": "synthetic/person-1/session-1/run-1",
                "trial_identifier": f"synthetic/{dataset_identifier}/trial-{epoch_index}",
                "start_sample": int(dataset_identifier) * 1000 + epoch_index * 32,
                "stop_sample": int(dataset_identifier) * 1000 + (epoch_index + 1) * 32,
            }
        ],
    }


def create_dataset(value, count=2):
    return EpochPreprocessedDTO(
        data=[
            RecordingDTO(
                data=numpy.full((count, 2, 32), value, dtype=numpy.float32),
                dataset_name="synthetic",
                subject_id=1,
                session_id="0",
                run_id="0",
                metadata={
                    "labels": numpy.arange(count) % 2,
                    "signal_specification": create_signal_specification(),
                    "epoch_origins": [create_epoch_origin(value, index) for index in range(count)],
                },
            )
        ]
    )
