from dataclasses import dataclass

from src.types.dto.config.raw_preprocessing_config import RawPreprocessingConfig
from src.types.dto.load.recording import RecordingDTO


@dataclass(frozen=True)
class RawPreprocessingInputDTO:
    """
    A data transfer object representing the raw input signal before any processing.

    This class is immutable to ensure data integrity as it passes through
    the transformation steps.

    Attributes:
        raw_preprocessing_config (RawPreprocessingConfig): The raw preprocessing config.
        signal (mne.io.Raw): The raw signal data to be preprocessed.
    """

    raw_preprocessing_config: RawPreprocessingConfig

    data: list[RecordingDTO]
