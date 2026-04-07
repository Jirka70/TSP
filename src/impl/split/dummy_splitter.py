import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.splitter import ISplitter


class DummySplitter(ISplitter):
    def run(
        self, input_dto: SplitInputDTO, run_ctx: RunContext
    ) -> StepResult[DatasetSplitDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy splitter")
        epochingData: EpochingDataDTO = EpochingDataDTO(
            data="",
            labels=[],
            event_names=[],
            sampling_rate_hz=12,
            n_epochs=2,
            n_channels=64,
            n_times=2,
            channel_names=[""],
        )
        data: DatasetSplitDTO = DatasetSplitDTO(
            epochingData, epochingData, epochingData
        )
        return StepResult(data)
