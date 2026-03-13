from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SplitConfig:

    enabled: bool
    backend: str

    train_ratio: float
    validation_ratio: float
    test_ratio: float

    shuffle: bool
    random_seed: int
