from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ValidationInput:
    """
    Validation step input
    """

    raw_config: dict[str, Any]
    source_name: str = "config.yaml"
