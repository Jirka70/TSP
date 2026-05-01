import logging
from pathlib import Path

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline.pipeline import IPipeline
from src.pipeline.run_context_factory import RunContextFactory
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.fold_evaluation_result_dto import FoldEvaluationResultDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.model.trained_model_dto import TrainedModelDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO
from src.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.artifact_saver import IArtifactSaver
from src.types.interfaces.augmentor import IAugmentor
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from src.types.interfaces.evaluator import IEvaluator
from src.types.interfaces.metrics_aggregator import IMetricsAggregator
from src.types.interfaces.model.final_trainer import IFinalTrainer
from src.types.interfaces.model.model_serializer import IModelSerializer
from src.types.interfaces.model.model_trainer import IModelTrainer
from src.types.interfaces.paradigm import IParadigm
from src.types.interfaces.raw_preprocessing import IRawPreprocessing
from src.types.interfaces.splitter import ISplitter
from src.types.interfaces.visualizer import IVisualizer


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
        metrics_aggregator: IMetricsAggregator,
        final_trainer: IFinalTrainer,
        evaluator: IEvaluator,
        visualizer: IVisualizer,
        artifact_saver: IArtifactSaver,
        model_serializer: IModelSerializer,
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
        self._metrics_aggregator = metrics_aggregator
        self._final_trainer = final_trainer
        self._evaluator = evaluator
        self._visualizer = visualizer
        self._artifact_saver = artifact_saver
        self._model_serializer = model_serializer

    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.source, run_ctx)

        raw_preprocessing_input: RawPreprocessingInputDTO = RawPreprocessingInputDTO(config.raw_preprocessing, load_result.data)
        raw_preprocessing_result: StepResult[RawPreprocessedDTO] = self._raw_preprocessing.run(raw_preprocessing_input, run_ctx)
        self._visualizer.visualize_raw(raw_preprocessing_result.data, run_ctx)

        paradigm_input: ParadigmInputDTO = ParadigmInputDTO(config.paradigm, raw_preprocessing_result.data)
        paradigm_result: StepResult[ParadigmResultDTO] = self._paradigm.run(paradigm_input, run_ctx)

        epoch_preprocessing_input: EpochPreprocessingInputDTO = EpochPreprocessingInputDTO(config.epoch_preprocessing, paradigm_result.data)
        epoch_preprocessing_result: StepResult[EpochPreprocessedDTO] = self._epoch_preprocessing.run(epoch_preprocessing_input, run_ctx)
        self._visualizer.visualize_epochs(epoch_preprocessing_result.data, run_ctx)

        splitting_input = SplitInputDTO(config.split, epoch_preprocessing_result.data)
        splitting_result = self._splitting.run(splitting_input, run_ctx)

        augmentation_input = AugmentationInputDTO(config.augmentation, splitting_result.data)
        augmentation_result = self._augmentation.run(augmentation_input, run_ctx)
        self._visualizer.visualize_augmentation(augmentation_result.data, run_ctx)

        folds = augmentation_result.data.folds
        if not folds:
            raise ValueError("Splitting/Augmentation returned no folds. Cannot continue training.")

        training_input = TrainingInputDTO(config=config.model, folds=folds, validation_data=augmentation_result.data.validation_data)
        model_training_result: StepResult[TrainingResultDTO] = self._model_trainer.run(training_input, run_ctx)

        metrics_input = TrainingResultDTO(model_training_result.data.trained_models)
        # Not using step result because it does not return anything (just log and future visualization)
        self._metrics_aggregator.run(metrics_input, run_ctx)

        final_trainer_input = FinalTrainingInputDTO(config=config.model, folds=folds, validation_data=augmentation_result.data.validation_data)
        final_training_result: StepResult[FinalTrainingResultDTO] = self._final_trainer.run(final_trainer_input, run_ctx)

        evaluation_input = EvaluationInputDTO(
            config=config.evaluation,
            trained_models=[final_training_result.data.trained_model],
            folds=folds,
        )
        evaluation_result = self._evaluator.run(evaluation_input, run_ctx)
        self._visualizer.visualize_evaluation(evaluation_result.data, run_ctx, final_training_result.data.trained_model.model_name)

        trained_model = final_training_result.data.trained_model
        save_artifacts_input: SaveArtifactsInputDTO = SaveArtifactsInputDTO(
            config.save_artifacts,
            config,
            output_path=Path("ahoj.txt"),
            evaluation_result=evaluation_result.data,
            trained_model=trained_model,
            model_serializer=self._model_serializer,
        )

        self._artifact_saver.run(save_artifacts_input, run_ctx)

    def _select_model_for_artifact_saving(
        self,
        trained_models: list[TrainedModelDTO],
        fold_results: list[FoldEvaluationResultDTO] | None,
    ) -> TrainedModelDTO:
        if not trained_models:
            raise ValueError("Training did not produce any model.")

        if not fold_results:
            return trained_models[0]

        best_fold = max(
            fold_results,
            key=lambda result: result.metrics.get("accuracy", float("-inf")),
        )
        for trained_model in trained_models:
            if trained_model.fold_idx == best_fold.fold_idx:
                return trained_model

        return trained_models[0]
