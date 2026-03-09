from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreprocessedDataDTO:
    signal: Any
    s_freq: float
    channel_names: list[str]
    metadata: dict[str, object]