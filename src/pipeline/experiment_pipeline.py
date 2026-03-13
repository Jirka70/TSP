import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline.run_context_factory import RunContextFactory
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.mode import Mode
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.epoching.epoching_input_dto import EpochingInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.preprocessing.preprocessed_data_dto import PreprocessedDataDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.augmentor import IAugmentor
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.epoching import IEpoching
from src.types.interfaces.preprocessing import IPreprocessing
from src.types.interfaces.splitter import ISplitter


class ExperimentPipeline:
    def __init__(
            self,
            data_loader: IDataLoader,
            preprocessing: IPreprocessing,
            epoching: IEpoching,
            splitting: ISplitter,
            augmentation: IAugmentor
    ) -> None:
        self._data_loader = data_loader
        self._run_context_factory = RunContextFactory()
        self._preprocessing = preprocessing
        self._epoching = epoching
        self._splitting = splitting
        self._log = logging.getLogger(__name__)
        self._augmentation = augmentation

    def run(self, config: ExperimentConfig) -> None:
        run_ctx: RunContext = self._run_context_factory.create(config, "test", "experiment_pipeline")
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.dataset, run_ctx)

        preprocessing_input: PreprocessingInputDTO = PreprocessingInputDTO(load_result.data, config.preprocessing)
        preprocessing_result: StepResult[PreprocessedDataDTO] = self._preprocessing.run(preprocessing_input, run_ctx)

        epoching_input: EpochingInputDTO = EpochingInputDTO(config.epoching, preprocessing_result.data)
        epoching_result: StepResult[EpochingDataDTO] = self._epoching.run(epoching_input, run_ctx)

        if config.mode == Mode.TRAINING.value:
            self._log.info("Preparing data for model training because mode was set to 'training'")

            splitting_input: SplitInputDTO = SplitInputDTO(config.split, epoching_result.data)
            splitting_result: StepResult[DatasetSplitDTO] = self._splitting.run(splitting_input, run_ctx)
            train_data = splitting_result.data.train_data # preparing training data for augmentation

            augmentation_input: AugmentationInputDTO = AugmentationInputDTO(config.augmentation, train_data)
            augmentation_result: StepResult[EpochingDataDTO] = self._augmentation.run(augmentation_input, run_ctx)


        elif config.mode == Mode.EXPERIMENT:
            pass
        else:
            pass
