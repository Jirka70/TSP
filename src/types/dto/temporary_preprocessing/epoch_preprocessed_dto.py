"""
This module defines the Data Transfer Objects (DTOs) for storing preprocessed epochs.

It ensures that the output of the epoch-based processing stage is encapsulated
in a consistent and immutable structure.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpochPreprocessedDTO:
    """
    A data transfer object representing the result of the epoch epoch_preprocessing stage.

    This DTO holds the processed signal data for a specific epoch (or set of epochs),
    ensuring that the data remains read-only and consistent for subsequent
    analysis or machine learning tasks.

    Attributes:
        signal (Any): The preprocessed signal data for the epoch(s).
    """

    signal: Any
