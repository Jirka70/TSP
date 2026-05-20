from typing import Literal

from tsp_eeg_classification.types.dto.config.astageconfig import AStageConfig


class FinalTrainerConfig(AStageConfig):
    """Configuration for the final model training stage."""

    backend: Literal["sklearn", "eegnet"]
