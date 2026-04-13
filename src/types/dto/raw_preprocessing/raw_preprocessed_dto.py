from dataclasses import dataclass

from src.types.dto.load.recording import RecordingDTO


@dataclass(frozen=True)
class RawPreprocessedDTO:
    """
    A data transfer object representing a signal after it has been preprocessed.

    This DTO acts as a read-only container for the output of the epoch_preprocessing
    logic, ensuring that the processed state remains consistent for downstream
    analysis or storage.

    Attributes:
        signal (mne.io.Raw): The preprocessed signal data, filtered and re-referenced.
    """

    data: list[RecordingDTO]
