from dataclasses import dataclass
from typing import Union

from pydantic import Field, BaseModel

from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.config.epoching_config import EpochingConfig
from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.config.mode import Mode
from src.types.dto.config.model.model_config import ModelConfig
from src.types.dto.config.preprocessing_config import PreprocessingConfigMNE
from src.types.dto.config.save_artifacts_config import SaveArtifactsConfig
from src.types.dto.config.split_config import SplitConfig
from src.types.dto.config.augmentation_config import AugmentationConfigBasic, AugmentationConfigNone


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
    preprocessing: Union[PreprocessingConfigMNE] = Field(discriminator="backend")

    epoching: Union[EpochingConfig] = Field(discriminator="backend")
    augmentation: Union[AugmentationConfigBasic, AugmentationConfigNone] = Field(discriminator="backend")
