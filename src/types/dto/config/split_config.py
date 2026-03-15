from typing import Literal

from pydantic import ImportString

from src.types.dto.config.astageconfig import AStageConfig


class SplitConfig(AStageConfig):
    _target_class = "impl.split.dummy_splitter.DummySplitter"
    backend: Literal["default"]
    enabled: bool

    train_ratio: float
    validation_ratio: float
    test_ratio: float

    shuffle: bool
    random_seed: int