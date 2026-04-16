from abc import ABC, abstractmethod

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO


class IDataLoader(ABC):
    @abstractmethod
    def run(self, input: ExternalDatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        raise NotImplementedError
