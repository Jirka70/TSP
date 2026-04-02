"""
This module defines the Data Transfer Objects (DTOs) for paradigm-specific preprocessing tasks.

It encapsulates the input structures required for
processing signals within a defined experimental or algorithmic paradigm.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParadigmPreprocessingInputDTO:
    """
    A data transfer object carrying the input signal for a paradigm-based preprocessing routine.

    This DTO is used to package the signal data before it is transformed
    according to the specific requirements of the current paradigm. Being
    immutable, it prevents any unintended modifications during the pipeline
    execution.

    Attributes:
        signal (Any): The input signal data associated with the paradigm.
    """

    signal: Any
