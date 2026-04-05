from dataclasses import dataclass
from typing import Any

from src.types.dto.config.epoch_preprocessing_config import EpochPreprocessingConfig


@dataclass(frozen=True)
class EpochPreprocessingInputDTO:
    """
    A data transfer object representing the input for epoch_preprocessing individual epochs.

    This class encapsulates a single epoch or a collection of epochs before
    they undergo specific transformations. Using a frozen dataclass ensures that
    the raw epoch data remains unaltered throughout the epoch_preprocessing stage.

    Attributes:
        epoch_preprocessing_config (EpochPreprocessingConfig): The configuration.
        signal (Any): The signal data corresponding to the epoch(s).
    """

    epoch_preprocessing_config: EpochPreprocessingConfig

    signal: Any
