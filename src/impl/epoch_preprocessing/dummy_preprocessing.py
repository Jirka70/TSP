import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.interfaces.preprocessing import IPreprocessing


class DummyPreprocessing(IPreprocessing):
    def run(self, input_dto: PreprocessingInputDTO, run_ctx: RunContext) -> StepResult[PreprocessedDataDTO]:
        log = logging.getLogger(__name__)
        log.info("Running dummy epoch_preprocessing")
        return StepResult(PreprocessedDataDTO(signal="",
                                              s_freq=128.0,
                                              channel_names=["channel1", "channel2", "channel3"]))
