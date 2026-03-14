from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


class SplitConfig(BaseModel):
    backend: Literal["default"]
    enabled: bool

    train_ratio: float
    validation_ratio: float
    test_ratio: float

    shuffle: bool
    random_seed: int
