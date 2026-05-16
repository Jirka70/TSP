from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedFold:
    fold_idx: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray | None
    y_test: np.ndarray | None