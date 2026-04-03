"""
This module contains Data Transfer Objects (DTOs) for storing processed data.

It facilitates the structured transfer of signals that have already undergone
initial epoch_preprocessing steps.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawPreprocessedDTO:
    """
    A data transfer object representing a signal after it has been preprocessed.

    This DTO acts as a read-only container for the output of the epoch_preprocessing
    logic, ensuring that the processed state remains consistent for downstream
    analysis or storage.

    Attributes:
        signal (Any): The preprocessed signal data, typically filtered or normalized.
    """

    signal: Any
