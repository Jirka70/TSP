"""
This module defines Data Transfer Objects (DTOs) for the epoch_preprocessing pipeline.

It ensures consistent data structures are passed between the ingestion and processing layers.
"""

from dataclasses import dataclass
from typing import Any

from src.types.dto.config.raw_preprocessing_config import RawPreprocessingConfig


@dataclass(frozen=True)
class RawPreprocessingInputDto:
    """
    A data transfer object representing the raw input signal before any processing.

    This class is immutable to ensure data integrity as it passes through
    the transformation steps.

    Attributes:
        signal (Any): The raw signal data to be preprocessed.
    """

    raw_preprocessing_config: RawPreprocessingConfig

    signal: Any
