from dataclasses import dataclass

from src.types.dto.config.model.model_config import EEGNetConfig, SklearnModelConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO


@dataclass(frozen=True)
class FinalTrainingInputDTO:
    """
    Input data for the final model training stage (training pipeline only).

    Attributes:
        config: Model configuration (ML or DL).
        training_data: Augmented training data used to fit the final model.
        validation_data: Validation data used for per-epoch evaluation during DL training.
            May be ``None`` for ML models or when ``validation_ratio`` is 0.0.
    """

    config: EEGNetConfig | SklearnModelConfig
    training_data: EpochPreprocessedDTO
    validation_data: EpochPreprocessedDTO | None = None
