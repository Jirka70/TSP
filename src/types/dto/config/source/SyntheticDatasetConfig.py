from typing import Literal
from src.types.dto.config.astageconfig import AStageConfig


class SyntheticDatasetConfig(AStageConfig):
    backend: Literal["synthetic"]
    s_freq: float = 128.0
    n_channels: int = 8
    n_trials: int = 24
    random_seed: int = 42