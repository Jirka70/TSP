import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline.run_context_factory import RunContextFactory
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.mode import Mode
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.epoching.epoching_input_dto import EpochingInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.interfaces.augmentor import IAugmentor
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.epoching import IEpoching
from src.types.interfaces.preprocessing import IPreprocessing


class ExperimentPipeline:
    def __init__(
            self,
            data_loader: IDataLoader,
            preprocessing: IPreprocessing,
            epoching: IEpoching
            # augmentation: IAugmentor
    ) -> None:
        self._data_loader = data_loader
        self._run_context_factory = RunContextFactory()
        self._preprocessing = preprocessing
        self._epoching = epoching
        # self._augmentation = augmentation
        self._log = logging.Logger(__name__)

    def run(self, config: ExperimentConfig) -> None:

        run_ctx: RunContext = self._run_context_factory.create(config, "test", "experiment_pipeline")
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.dataset, run_ctx)

        preprocessing_input: PreprocessingInputDTO = PreprocessingInputDTO(load_result.data, config.preprocessing)
        preprocessing_result: StepResult[PreprocessedDataDTO] = self._preprocessing.run(preprocessing_input, run_ctx)

        epoching_input: EpochingInputDTO = EpochingInputDTO(config.epoching, preprocessing_result.data)
        epoching_result: StepResult[EpochingDataDTO] = self._epoching.run(epoching_input, run_ctx)

        if config.mode == Mode.TRAINING.value:
            pass
            # self._augmentation.run()
        elif config.mode == Mode.EXPERIMENT:
            pass
        else:
            self._log.warning(f"Unknown mode: {config.mode}""")
