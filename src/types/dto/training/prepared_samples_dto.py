from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedSamplesDTO:
    """
    Output of sample preparation step for model
    """

    X: Any
    """
    Input data for model (numpy array, torch tensor). Type is not specified implicitly...
    """

    y: Any
    """
    Correct answer (answer from the "teacher")
    """

    metadata: dict[str, object]
    """
    Additional info about dataset (optional)
    """