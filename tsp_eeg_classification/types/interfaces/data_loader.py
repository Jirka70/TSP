from abc import ABC, abstractmethod

from tsp_eeg_classification.pipeline.context.run_context import RunContext
from tsp_eeg_classification.pipeline.contracts.step_result import StepResult
from tsp_eeg_classification.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
from tsp_eeg_classification.types.dto.config.source.filesystem_dataset_config import FilesystemDatasetConfig
from tsp_eeg_classification.types.dto.load.raw_data_dto import RawDataDTO


class IDataLoader(ABC):
    @abstractmethod
    def run(self, input: ExternalDatasetConfig | FilesystemDatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        raise NotImplementedError
