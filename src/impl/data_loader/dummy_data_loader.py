import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.interfaces.data_loader import IDataLoader


class DummyLoader(IDataLoader):
    def run(self, input_dto: DatasetConfig, run_ctx: RunContext) -> StepResult[RawDataDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy data loader")
        data = RawDataDTO(sampling_freq=42, channel_names=[], signal="")
        return StepResult(data, None, [])