from typing import Literal

from src.types.dto.config.astageconfig import AStageConfig


class SplitConfig(AStageConfig):
    backend: Literal["default"]
    enabled: bool

    train_ratio: float
    validation_ratio: float
    test_ratio: float

    shuffle: bool
    random_seed: int
