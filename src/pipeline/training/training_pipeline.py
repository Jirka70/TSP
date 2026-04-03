import logging
from pathlib import Path

from src.impl.model.pytorch_serializer import PyTorchSerializer
from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline.pipeline import IPipeline
from src.pipeline.run_context_factory import RunContextFactory
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.dto.epoching.epoching_data_dto import EpochingDataDTO
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.dto.paradigm.paradigm_preprocessed_dto import ParadigmPreprocessedDTO
from src.types.dto.paradigm.paradigm_preprocessing_input_dto import ParadigmPreprocessingInputDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDto
from src.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.artifact_saver import IArtifactSaver
from src.types.interfaces.augmentor import IAugmentor
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from src.types.interfaces.evaluator import IEvaluator
from src.types.interfaces.model.model_trainer import IModelTrainer
from src.types.interfaces.paradigm import IParadigm
from src.types.interfaces.raw_preprocessing import IRawPreprocessing
from src.types.interfaces.splitter import ISplitter


class TrainingPipeline(IPipeline):
    def __init__(
        self,
        data_loader: IDataLoader,
        raw_preprocessing: IRawPreprocessing,
        paradigm: IParadigm,
        epoch_preprocessing: IEpochPreprocessing,
        splitting: ISplitter,
        augmentation: IAugmentor,
        model_trainer: IModelTrainer,
        evaluator: IEvaluator,
        artifact_saver: IArtifactSaver,
    ) -> None:
        self._data_loader = data_loader
        self._run_context_factory = RunContextFactory()
        self._raw_preprocessing = raw_preprocessing
        self._paradigm = paradigm
        self._epoch_preprocessing = epoch_preprocessing
        self._splitting = splitting
        self._log = logging.getLogger(__name__)
        self._augmentation = augmentation
        self._model_trainer = model_trainer
        self._evaluator = evaluator
        self._artifact_saver = artifact_saver

    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.dataset, run_ctx)

        raw_preprocessing_input: RawPreprocessingInputDto = RawPreprocessingInputDto(config.raw_preprocessing, load_result.data)
        raw_preprocessing_result: StepResult[RawPreprocessedDTO] = self._raw_preprocessing.run(raw_preprocessing_input, run_ctx)

        paradigm_input: ParadigmPreprocessingInputDTO = ParadigmPreprocessingInputDTO(config.paradigm, raw_preprocessing_result.data)
        paradigm_result: StepResult[ParadigmPreprocessedDTO] = self._paradigm.run(paradigm_input, run_ctx)

        epoch_preprocessing_input: EpochPreprocessingInputDTO = EpochPreprocessingInputDTO(config.epoch_preprocessing, paradigm_result.data)
        epoch_preprocessing_result: StepResult[EpochPreprocessedDTO] = self._epoch_preprocessing.run(epoch_preprocessing_input, run_ctx)

        splitting_input: SplitInputDTO = SplitInputDTO(config.split, epoch_preprocessing_result.data)
        splitting_result: StepResult[DatasetSplitDTO] = self._splitting.run(splitting_input, run_ctx)
        train_data: EpochingDataDTO = splitting_result.data.train_data  # preparing training data for augmentation

        augmentation_input: AugmentationInputDTO = AugmentationInputDTO(config.augmentation, train_data)
        augmentation_result: StepResult[EpochingDataDTO] = self._augmentation.run(augmentation_input, run_ctx)

        validation_data: EpochingDataDTO = splitting_result.data.validation_data
        training_input: TrainingInputDTO = TrainingInputDTO(config.model, augmentation_result.data, validation_data)
        model_training_result: StepResult[TrainedModelDTO] = self._model_trainer.run(training_input, run_ctx)

        test_data: EpochingDataDTO = splitting_result.data.test_data
        evaluation_input: EvaluationInputDTO = EvaluationInputDTO(config.evaluation, model_training_result.data, test_data)
        evaluation_result: StepResult[EvaluationResultDTO] = self._evaluator.run(evaluation_input, run_ctx)

        trained_model = model_training_result.data
        save_artifacts_input: SaveArtifactsInputDTO = SaveArtifactsInputDTO(
            config.save_artifacts,
            config,
            output_path=Path("ahoj.txt"),
            evaluation_result=evaluation_result.data,
            trained_model=trained_model,
            model_serializer=PyTorchSerializer(),
        )
        self._artifact_saver.run(save_artifacts_input, run_ctx)
