from dataclasses import dataclass

from src.types.dto.load.recording import RecordingDTO


@dataclass(frozen=True)
class EpochPreprocessedDTO:
    """
    A data transfer object representing the result of the epoch epoch_preprocessing stage.

    This DTO holds the processed signal data for a specific epoch (or set of epochs),
    ensuring that the data remains read-only and consistent for subsequent
    analysis or machine learning tasks.

    Attributes:
        data (list[RecordingDTO]): Preprocessed recordings; each entry holds cleaned MNE Epochs.
    """

    data: list[RecordingDTO]
