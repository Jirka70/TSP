from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader


class EDFLoader(IDataLoader):
    def run(self, input: ExternalDatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        # TODO implement method
        raise NotImplementedError
