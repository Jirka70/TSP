from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreparedSamplesDTO:

    X: Any
    """
    Input data for training (numpy array, torch tensor). Type is not specified implicitly...
    """

    y: Any
    """
    Correct answer (answer from the "teacher")
    """

    metadata: dict[str, object] | None = None
    """
    Additional info about dataset (optional)
    """