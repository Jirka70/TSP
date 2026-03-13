import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.epoching.epoching_input_dto import EpochingInputDTO
from src.types.interfaces.epoching import IEpoching


class DummyEpoching(IEpoching):
    def run(self, input_dto: EpochingInputDTO, run_ctx: RunContext) -> StepResult[EpochingDataDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy epoching")
        data: EpochingDataDTO = EpochingDataDTO(data="",
                                                labels=[],
                                                event_names=[],
                                                sampling_rate_hz=128,
                                                n_epochs=2,
                                                n_channels=1,
                                                n_times=2,
                                                channel_names=[]
                                                )

        return StepResult(data=data)
