from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from src.types.dto.config.augmentation_config import (
    AugmentationConfigBasic,
    AugmentationConfigNone,
    AugmentationConfigTorchEEG,
)
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.config.epoching_config import EpochingConfig
from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.config.model.model_config import ModelConfig
from src.types.dto.config.preprocessing_config import PreprocessingConfigMNE
from src.types.dto.config.save_artifacts_config import SaveArtifactsConfig
from src.types.dto.config.split_config import SplitConfig


class Mode(str, Enum):
    TRAINING = "training"
    EXPERIMENT = "experiment"


@dataclass
class ExperimentConfig(BaseModel):
    mode: Mode
    output_dir: str
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    dataset: DatasetConfig
    save_artifacts: SaveArtifactsConfig

    # union enables multiple options which pydantic differentiates by looking at backend field
    # for example: Union[PreprocessingConfigMNE, ProprocessingConfigMoabb, ...] = Field(dicriminator="backend")
    preprocessing: PreprocessingConfigMNE = Field(discriminator="backend")

    epoching: EpochingConfig = Field(discriminator="backend")
    augmentation: AugmentationConfigBasic | AugmentationConfigTorchEEG | AugmentationConfigNone = Field(discriminator="backend")
