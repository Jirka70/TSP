from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline.pipeline import IPipeline
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.epoching.epoching_input_dto import EpochingInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.epoching import IEpoching
from src.types.interfaces.preprocessing import IPreprocessing


class ExperimentPipeline(IPipeline):

    def __init__(
            self,
            data_loader: IDataLoader,
            preprocessing: IPreprocessing,
            epoching: IEpoching):
        self._data_loader = data_loader
        self._preprocessing = preprocessing
        self._epoching = epoching

    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.dataset, run_ctx)

        preprocessing_input: PreprocessingInputDTO = PreprocessingInputDTO(load_result.data, config.preprocessing)
        preprocessing_result: StepResult[PreprocessedDataDTO] = self._preprocessing.run(preprocessing_input, run_ctx)

        epoching_input: EpochingInputDTO = EpochingInputDTO(config.epoching, preprocessing_result.data)
        epoching_result: StepResult[EpochingDataDTO] = self._epoching.run(epoching_input, run_ctx)
