from typing import Union

from pydantic import Field

from src.types.dto.config.astageconfig import AStageConfig
from src.types.dto.config.experiment_config import Mode
from src.types.dto.config.preprocessing_config import PreprocessingConfigMNE


class RootConfig(AStageConfig):
    mode: Mode
    output_dir: str

    preprocessing: Union[PreprocessingConfigMNE] = Field(discriminator="backend")
    '''
    epoching: default
    split: default
    augmentation: none
    model: eegnet
    evaluation: default
    visualization: matplotlib
    dataset: eegbci
    '''
