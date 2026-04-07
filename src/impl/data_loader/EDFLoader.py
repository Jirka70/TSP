from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader


class EDFLoader(IDataLoader):
    def run(self, input: DatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        # TODO implement method
        raise NotImplementedError
