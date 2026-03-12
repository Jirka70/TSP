import logging

from src.pipeline.context.run_context import RunContext
from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.interfaces.preprocessing import IPreprocessing


class DummyPreprocessing(IPreprocessing):
    def run(self, input_dto: PreprocessingInputDTO, run_ctx: RunContext) -> PreprocessedDataDTO:
        log = logging.getLogger(__name__)
        log.info("Running dummy preprocessing")
        return PreprocessedDataDTO(signal="",
                                   s_freq=128.0,
                                   channel_names=["channel1", "channel2", "channel3"])
