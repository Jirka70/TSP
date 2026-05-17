from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO

import numpy as np


def extract_learning_data(
        data_dto: EpochPreprocessedDTO,
) -> tuple[np.ndarray, np.ndarray]:
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

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    if x.ndim != 3:
        raise ValueError(
            f"EEGNet expects input shape "
            f"(n_epochs, n_channels, n_times), got {x.shape}"
        )

    return x, y