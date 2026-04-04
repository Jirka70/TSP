from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline.pipeline import IPipeline
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.paradigm.paradigm_preprocessed_dto import ParadigmPreprocessedDTO
from src.types.dto.paradigm.paradigm_preprocessing_input_dto import ParadigmPreprocessingInputDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDto
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from src.types.interfaces.paradigm import IParadigm
from src.types.interfaces.raw_preprocessing import IRawPreprocessing


class ExperimentPipeline(IPipeline):
    def __init__(
        self,
        data_loader: IDataLoader,
        raw_preprocessing: IRawPreprocessing,
        paradigm: IParadigm,
        epoch_preprocessing: IEpochPreprocessing,
    ):
        self._data_loader = data_loader
        self._raw_preprocessing = raw_preprocessing
        self._paradigm = paradigm
        self._epoch_preprocessing = epoch_preprocessing

    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.dataset, run_ctx)

        raw_preprocessing_input: RawPreprocessingInputDto = RawPreprocessingInputDto(raw_preprocessing_config=config.raw_preprocessing, signal=load_result.data.signal)
        raw_preprocessing_result: StepResult[RawPreprocessedDTO] = self._raw_preprocessing.run(raw_preprocessing_input, run_ctx)

        paradigm_input: ParadigmPreprocessingInputDTO = ParadigmPreprocessingInputDTO(config.paradigm, raw_preprocessing_result.data)
        paradigm_result: StepResult[ParadigmPreprocessedDTO] = self._paradigm.run(paradigm_input, run_ctx)

        epoch_preprocessing_input: EpochPreprocessingInputDTO = EpochPreprocessingInputDTO(config.epoch_preprocessing, paradigm_result.data)
        epoch_preprocessing_result: StepResult[EpochPreprocessedDTO] = self._epoch_preprocessing.run(epoch_preprocessing_input, run_ctx)
