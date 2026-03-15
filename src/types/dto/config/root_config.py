from typing import Union

from src.types.dto.config.mode import Mode
from pydantic import Field

from src.types.dto.config.preprocessing_config import PreprocessingConfigMNE
from src.types.dto.config.iconfig import AStageConfig


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
