"""Small signal datasets containing only the labels required for training."""

import numpy

from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO


def create_dataset(value, count=2):
    return EpochPreprocessedDTO(
        data=[
            RecordingDTO(
                data=numpy.full((count, 2, 32), value, dtype=numpy.float32),
                dataset_name="synthetic",
                subject_id=1,
                session_id="0",
                run_id="0",
                metadata={"labels": numpy.arange(count) % 2},
            )
        ]
    )
