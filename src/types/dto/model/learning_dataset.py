from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LearningDataset:
    x: np.ndarray
    y: np.ndarray

    @property
    def sample_count(self) -> int:
        return len(self.y)