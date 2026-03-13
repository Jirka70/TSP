from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:

    enabled: bool
    backend: str

    train_ratio: float
    validation_ratio: float
    test_ratio: float

    shuffle: bool
    random_seed: int
