"""
This module provides Data Transfer Objects (DTOs) for storing the results of  paradigm-specific preprocessing.

It ensures that processed signals are maintained in a structured, immutable format for further consumption.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParadigmPreprocessedDTO:
    """
    A data transfer object representing the output of a paradigm-specific preprocessing stage.

    This class serves as a finalized container for signals that have been
    tailored to a specific experimental or analytical paradigm. The frozen
    state guarantees that the preprocessed results remain constant.

    Attributes:
        signal (Any): The preprocessed signal data relevant to the paradigm.
    """

    signal: Any
