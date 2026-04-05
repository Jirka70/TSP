from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParadigmResultDTO:
    """
    A data transfer object representing the output of a paradigm-specific epoch_preprocessing stage.

    This class serves as a finalized container for signals that have been
    tailored to a specific experimental or analytical paradigm. The frozen
    state guarantees that the preprocessed results remain constant.

    Attributes:
        signal (Any): The preprocessed signal data relevant to the paradigm.
    """

    signal: Any
