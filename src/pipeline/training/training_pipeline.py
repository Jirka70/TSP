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
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.model.final_training_result_dto import FinalTrainingResultDTO
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO
from src.types.dto.raw_augmentation.raw_augmentation_input_dto import RawAugmentationInputDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO
from src.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.artifact_saver import IArtifactSaver
from src.types.interfaces.augmentor import IAugmentor
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.dataset_exporter import IDatasetExporter
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from src.types.interfaces.evaluator import IEvaluator
from src.types.interfaces.model.final_trainer import IFinalTrainer
from src.types.interfaces.model.model_serializer import IModelSerializer
from src.types.interfaces.paradigm import IParadigm
from src.types.interfaces.raw_augmentor import IRawAugmentor
from src.types.interfaces.raw_preprocessing import IRawPreprocessing
from src.types.interfaces.splitter import ISplitter
from src.types.interfaces.visualizer import IVisualizer


class TrainingPipeline(IPipeline):
    """
    Orchestrator pipeline for training a final ML or DL model.

    Executes the following stages in order:
        1. Data loading
        2. Raw preprocessing + visualization
        3. Raw augmentation + visualization
        4. Paradigm extraction (epoching)
        5. Epoch preprocessing + visualization
        6. Basic splitting (train / test / validation; no cross-validation folds)
        7. Augmentation of train data + visualization
        8. Dataset export
        9. Final model training (on train data; validation used for DL per-epoch eval)
        10. Final model evaluation on held-out test data (if available)
        11. Artifact saving
    """

    def __init__(
        self,
        data_loader: IDataLoader,
        raw_preprocessing: IRawPreprocessing,
        raw_augmentation: IRawAugmentor,
        paradigm: IParadigm,
        epoch_preprocessing: IEpochPreprocessing,
        splitting: ISplitter,
        augmentation: IAugmentor,
        dataset_exporter: IDatasetExporter,
        final_trainer: IFinalTrainer,
        evaluator: IEvaluator,
        visualizer: IVisualizer,
        artifact_saver: IArtifactSaver,
        model_serializer: IModelSerializer,
    ) -> None:
        self._data_loader = data_loader
        self._run_context_factory = RunContextFactory()
        self._raw_preprocessing = raw_preprocessing
        self._raw_augmentation = raw_augmentation
        self._paradigm = paradigm
        self._epoch_preprocessing = epoch_preprocessing
        self._splitting = splitting
        self._log = logging.getLogger(__name__)
        self._augmentation = augmentation
        self._dataset_exporter = dataset_exporter
        self._final_trainer = final_trainer
        self._evaluator = evaluator
        self._visualizer = visualizer
        self._artifact_saver = artifact_saver
        self._model_serializer = model_serializer

    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        """
        Execute the training pipeline sequentially.

        Args:
            config (ExperimentConfig): The master experiment configuration.
            run_ctx (RunContext): Context keeping track of unique runtime metadata.

        Raises:
            ValueError: If splitting produces no train data.
        """
        # --- 1. Data Loading ---
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.source, run_ctx)

        # --- 2. Raw Preprocessing ---
        raw_preprocessing_input: RawPreprocessingInputDTO = RawPreprocessingInputDTO(config.raw_preprocessing, load_result.data)
        raw_preprocessing_result: StepResult[RawPreprocessedDTO] = self._raw_preprocessing.run(raw_preprocessing_input, run_ctx)
        self._visualizer.visualize_raw(raw_preprocessing_result.data, run_ctx)

        # --- 3. Raw Augmentation ---
        raw_augmentation_input = RawAugmentationInputDTO(config.raw_augmentation, raw_preprocessing_result.data)
        raw_augmentation_result = self._raw_augmentation.run(raw_augmentation_input, run_ctx)

        copies_per_sample_raw = getattr(config.raw_augmentation, "copies_per_sample", 0)
        self._visualizer.visualize_raw_augmentation(raw_augmentation_result.data, run_ctx, copies_per_sample_raw)

        # --- 4. Paradigm Extraction ---
        paradigm_input: ParadigmInputDTO = ParadigmInputDTO(config.paradigm, raw_augmentation_result.data)
        paradigm_result: StepResult[ParadigmResultDTO] = self._paradigm.run(paradigm_input, run_ctx)

        # --- 5. Epoch Preprocessing ---
        epoch_preprocessing_input: EpochPreprocessingInputDTO = EpochPreprocessingInputDTO(config.epoch_preprocessing, paradigm_result.data)
        epoch_preprocessing_result: StepResult[EpochPreprocessedDTO] = self._epoch_preprocessing.run(epoch_preprocessing_input, run_ctx)
        self._visualizer.visualize_epochs(epoch_preprocessing_result.data, run_ctx)

        # --- 6. Splitting (train / test / validation; folds=None) ---
        splitting_input = SplitInputDTO(config.split, epoch_preprocessing_result.data)
        splitting_result: StepResult[DatasetSplitDTO] = self._splitting.run(splitting_input, run_ctx)

        if splitting_result.data.train is None:
            raise ValueError("Splitting returned no train data. Cannot continue training.")

        # --- 7. Augmentation (train data only) ---
        augmentation_input = AugmentationInputDTO(config.augmentation, splitting_result.data)
        augmentation_result: StepResult[DatasetSplitDTO] = self._augmentation.run(augmentation_input, run_ctx)

        copies_per_sample = getattr(config.augmentation, "copies_per_sample", 0)
        self._visualizer.visualize_augmentation(augmentation_result.data, run_ctx, copies_per_sample)

        # --- 8. Dataset Export ---
        self._dataset_exporter.run(config.dataset_export, augmentation_result.data, run_ctx)

        # --- 9. Final Model Training ---
        # TODO There are no folds!!! EDIT YOUR WORK TO NOT USE IT!
        augmented_split: DatasetSplitDTO = augmentation_result.data
        final_trainer_input = FinalTrainingInputDTO(
            config=config.model,
            training_data=augmented_split.train,
            validation_data=augmented_split.validation,
        )
        final_training_result: StepResult[FinalTrainingResultDTO] = self._final_trainer.run(final_trainer_input, run_ctx)

        # --- 10. Final Model Evaluation (on held-out test data, if available) ---
        if augmented_split.test is not None:
            evaluation_input = EvaluationInputDTO(
                config=config.evaluation,
                trained_model=final_training_result.data.trained_model,
                test_data=augmented_split.test,
            )
            evaluation_result = self._evaluator.run(evaluation_input, run_ctx)
            self._visualizer.visualize_evaluation(evaluation_result.data, run_ctx, final_training_result.data.trained_model.model_name)
        else:
            self._log.warning("No test data available (test_ratio=0.0). Skipping final model evaluation.")
            evaluation_result = None

        # --- 11. Artifact Saving ---
        trained_model = final_training_result.data.trained_model
        save_artifacts_input: SaveArtifactsInputDTO = SaveArtifactsInputDTO(
            config.save_artifacts,
            config,
            output_path=Path("model_output"),
            evaluation_result=evaluation_result.data if evaluation_result is not None else None,
            trained_model=trained_model,
            model_serializer=self._model_serializer,
        )
        self._artifact_saver.run(save_artifacts_input, run_ctx)
