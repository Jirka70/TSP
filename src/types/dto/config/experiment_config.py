from dataclasses import dataclass

from src.types.dto.config.augmentation_config import AugmentationConfig
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.config.epoching_config import EpochingConfig
from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.config.mode import Mode
from src.types.dto.config.model_config import ModelConfig
from src.types.dto.config.preprocessing_config import PreprocessingConfig
from src.types.dto.config.split_config import SplitConfig


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    epoching: EpochingConfig
    split: SplitConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    mode: Mode
