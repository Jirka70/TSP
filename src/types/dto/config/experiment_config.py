from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from src.types.dto.config.augmentation_config import AugmentationConfigBasic, AugmentationConfigNone
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.config.epoch_preprocessing_config import EpochPreprocessingConfig
from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.config.model.model_config import ModelConfig
from src.types.dto.config.paradigm_config import ParadigmConfig
from src.types.dto.config.raw_preprocessing_config import RawPreprocessingConfig
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
    # for example: Union[PreprocessingConfigMNE, ProprocessingConfigMoabb, ...] = Field(discriminator="backend")
    raw_preprocessing: RawPreprocessingConfig = Field(discriminator="backend")
    paradigm: ParadigmConfig = Field(discriminator="backend")
    epoch_preprocessing: EpochPreprocessingConfig = Field(discriminator="backend")

    augmentation: AugmentationConfigBasic | AugmentationConfigNone = Field(discriminator="backend")
