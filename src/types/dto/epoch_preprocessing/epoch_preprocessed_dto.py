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
        signal (np.ndarray): The preprocessed signal data for the epoch(s), shaped
            (n_epochs, n_csp_components) after CSP transformation.
        labels (np.ndarray): Integer class label for each epoch, shaped (n_epochs,).
    """

    data: list[RecordingDTO]
