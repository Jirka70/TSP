from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from tsp_eeg_classification.types.dto.config.augmentation_config import (
    AugmentationConfigBasic,
    AugmentationConfigNone,
    AugmentationConfigTorchEEG,
)
from tsp_eeg_classification.types.dto.config.logging.logging_config import LoggingConfig
from tsp_eeg_classification.types.dto.config.model.model_path_config import ModelPathConfig
from tsp_eeg_classification.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
from tsp_eeg_classification.types.dto.config.dataset_export_config import DatasetExportConfig
from tsp_eeg_classification.types.dto.config.epoch_preprocessing_config import EpochPreprocessingConfig
from tsp_eeg_classification.types.dto.config.evaluation_config import EvaluationConfig, SklearnEvaluationConfig
from tsp_eeg_classification.types.dto.config.model.final_trainer_config import FinalTrainerConfig
from tsp_eeg_classification.types.dto.config.model.metrics_aggregator_config import MetricsAggregatorConfig
from tsp_eeg_classification.types.dto.config.model.model_config import EEGNetConfig, SklearnModelConfig
from tsp_eeg_classification.types.dto.config.model.model_path_config import ModelPathConfig
from tsp_eeg_classification.types.dto.config.paradigm_config import ParadigmConfig
from tsp_eeg_classification.types.dto.config.raw_augmentation_config import (
    RawAugmentationConfigNone,
    RawAugmentationConfigTorchEEG,
)
from tsp_eeg_classification.types.dto.config.raw_preprocessing_config import RawPreprocessingConfig
from tsp_eeg_classification.types.dto.config.save_artifacts_config import SaveArtifactsConfig
from tsp_eeg_classification.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
from tsp_eeg_classification.types.dto.config.source.filesystem_dataset_config import FilesystemDatasetConfig
from tsp_eeg_classification.types.dto.config.split_config import SplitConfig, SplitMoabbCrossSessionConfig, SplitMoabbCrossSubjectConfig, SplitMoabbWithinSessionConfig, SplitMoabbWithinSubjectConfig
from tsp_eeg_classification.types.dto.config.visualization_config import VisualizationConfig


class Mode(str, Enum):
    TRAINING = "training"
    EXPERIMENT = "experiment"


@dataclass
class ExperimentConfig(BaseModel):
    mode: Mode
    output_dir: str
    logging: LoggingConfig
    save_artifacts: SaveArtifactsConfig
    metrics_aggregator: MetricsAggregatorConfig
    final_trainer: FinalTrainerConfig
    model_path: ModelPathConfig

    # union enables multiple options which pydantic differentiates by looking at backend field
    # for example: Union[PreprocessingConfigMNE, ProprocessingConfigMoabb, ...] = Field(discriminator="backend")
    model: EEGNetConfig | SklearnModelConfig = Field(discriminator="backend")
    evaluation: EvaluationConfig | SklearnEvaluationConfig = Field(discriminator="backend")
    raw_preprocessing: RawPreprocessingConfig = Field(discriminator="backend")
    raw_augmentation: RawAugmentationConfigNone | RawAugmentationConfigTorchEEG = Field(discriminator="backend")
    paradigm: ParadigmConfig = Field(discriminator="backend")
    epoch_preprocessing: EpochPreprocessingConfig = Field(discriminator="backend")
    split: SplitConfig | SplitMoabbWithinSessionConfig | SplitMoabbWithinSubjectConfig | SplitMoabbCrossSessionConfig | SplitMoabbCrossSubjectConfig = Field(discriminator="backend")
    source: FilesystemDatasetConfig | ExternalDatasetConfig = Field(discriminator="backend")
    augmentation: AugmentationConfigBasic | AugmentationConfigTorchEEG | AugmentationConfigNone = Field(discriminator="backend")
    visualization: VisualizationConfig = Field(discriminator="backend")
    dataset_export: DatasetExportConfig = Field(discriminator="backend")
