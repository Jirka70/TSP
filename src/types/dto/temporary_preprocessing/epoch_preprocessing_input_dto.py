"""
This module contains Data Transfer Objects (DTOs) for epoch-based epoch_preprocessing.

It defines the structures required to feed segmented signal data (epochs) into
the processing pipeline.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpochPreprocessingInputDTO:
    """
    A data transfer object representing the input for epoch_preprocessing individual epochs.

    This class encapsulates a single epoch or a collection of epochs before
    they undergo specific transformations. Using a frozen dataclass ensures that
    the raw epoch data remains unaltered throughout the epoch_preprocessing stage.

    Attributes:
        signal (Any): The signal data corresponding to the epoch(s).
    """

    signal: Any
