from dataclasses import dataclass
from typing import Any

from src.types.dto.config.raw_preprocessing_config import RawPreprocessingConfig


@dataclass(frozen=True)
class RawPreprocessingInputDTO:
    """
    A data transfer object representing the raw input signal before any processing.

    This class is immutable to ensure data integrity as it passes through
    the transformation steps.

    Attributes:
        raw_preprocessing_config (RawPreprocessingConfig): The raw preprocessing config.
        signal (Any): The raw signal data to be preprocessed.
    """

    raw_preprocessing_config: RawPreprocessingConfig

    signal: Any
