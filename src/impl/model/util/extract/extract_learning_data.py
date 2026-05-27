from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.learning_dataset import LearningDataset

import numpy as np


def extract_learning_data(
        data_dto: EpochPreprocessedDTO,
) -> LearningDataset:
    x_list = []
    y_list = []

    for recording in data_dto.data:
        epochs = recording.data

        if hasattr(epochs, "get_data"):
            x_list.append(epochs.get_data(copy=False))
            y_list.append(epochs.events[:, -1])
        else:
            x_list.append(epochs)
            y_list.append(np.array(recording.metadata.get("labels", [])))

    if not x_list or not y_list:
        raise ValueError("No learning data found. Expected at least one recording.")

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if len(x) != len(y):
        raise ValueError(
            f"Learning data and labels have different sample counts: "
            f"x={len(x)}, y={len(y)}"
        )

    if x.ndim != 3:
        raise ValueError(
            f"EEGNet expects input shape "
            f"(n_epochs, n_channels, n_times), got {x.shape}"
        )

    return LearningDataset(x=x, y=y)
