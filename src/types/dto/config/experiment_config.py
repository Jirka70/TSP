from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from src.types.dto.config.final_trainer_config import FinalTrainerConfig
from src.types.dto.config.augmentation_config import (
    AugmentationConfigBasic,
    AugmentationConfigNone,
    AugmentationConfigTorchEEG,
)
from src.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
from src.types.dto.config.epoch_preprocessing_config import EpochPreprocessingConfig
from src.types.dto.config.evaluation_config import EvaluationConfig, SklearnEvaluationConfig
from src.types.dto.config.metrics_aggregator_config import MetricsAggregatorConfig
from src.types.dto.config.model.model_config import ModelConfig, SklearnModelConfig
from src.types.dto.config.paradigm_config import ParadigmConfig
from src.types.dto.config.raw_preprocessing_config import RawPreprocessingConfig
from src.types.dto.config.save_artifacts_config import SaveArtifactsConfig
from src.types.dto.config.source.filesystem_dataset_config import FilesystemDatasetConfig
from src.types.dto.config.split_config import SplitConfig, SplitMoabbCrossSessionConfig, SplitMoabbCrossSubjectConfig, SplitMoabbWithinSessionConfig, SplitMoabbWithinSubjectConfig


class Mode(str, Enum):
    TRAINING = "training"
    EXPERIMENT = "experiment"


@dataclass
class ExperimentConfig(BaseModel):
    mode: Mode
    output_dir: str
    save_artifacts: SaveArtifactsConfig
    metrics_aggregator: MetricsAggregatorConfig
    final_trainer: FinalTrainerConfig

    # union enables multiple options which pydantic differentiates by looking at backend field
    # for example: Union[PreprocessingConfigMNE, ProprocessingConfigMoabb, ...] = Field(discriminator="backend")
    model: ModelConfig | SklearnModelConfig = Field(discriminator="backend")
    evaluation: EvaluationConfig | SklearnEvaluationConfig = Field(discriminator="backend")
    raw_preprocessing: RawPreprocessingConfig = Field(discriminator="backend")
    paradigm: ParadigmConfig = Field(discriminator="backend")
    epoch_preprocessing: EpochPreprocessingConfig = Field(discriminator="backend")
    split: SplitConfig | SplitMoabbWithinSessionConfig | SplitMoabbWithinSubjectConfig | SplitMoabbCrossSessionConfig | SplitMoabbCrossSubjectConfig = Field(discriminator="backend")
    source: FilesystemDatasetConfig | ExternalDatasetConfig = Field(discriminator="backend")
    augmentation: AugmentationConfigBasic | AugmentationConfigTorchEEG | AugmentationConfigNone = Field(discriminator="backend")
