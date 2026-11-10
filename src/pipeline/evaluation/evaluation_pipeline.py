"""Evaluation pipeline for cross-validation feasibility assessment."""

import logging

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.pipeline.pipeline import IPipeline
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.dto.evaluation.evaluation_input_dto import CrossValidationEvaluationInputDTO
from src.types.dto.load.raw_data_dto import RawDataDTO
from src.types.dto.model.training_input_dto import TrainingInputDTO
from src.types.dto.model.training_result_dto import TrainingResultDTO
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.paradigm.paradigm_result_dto import ParadigmResultDTO
from src.types.dto.raw_augmentation.raw_augmentation_input_dto import RawAugmentationInputDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.types.interfaces.augmentor import IAugmentor
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.dataset_exporter import IDatasetExporter
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from src.types.interfaces.evaluator import ICrossValidationEvaluator
from src.types.interfaces.model.model_trainer import IModelTrainer
from src.types.interfaces.paradigm import IParadigm
from src.types.interfaces.raw_augmentor import IRawAugmentor
from src.types.interfaces.raw_preprocessing import IRawPreprocessing
from src.types.interfaces.splitter import ISplitter
from src.types.interfaces.visualizer import IVisualizer


class EvaluationPipeline(IPipeline):
    """
    Orchestrator pipeline for cross-validation feasibility assessment.

    Executes the same preprocessing stages as TrainingPipeline (data loading,
    raw preprocessing, raw augmentation, paradigm extraction, epoch preprocessing,
    MOABB-based splitting, augmentation, and dataset export), but instead of
    training a large final model, it only trains small cross-validated ML models
    (via model_trainer) and then evaluates each model on its fold's test data
    (via a fold-level evaluator).

    This pipeline is designed to quickly assess whether a given dataset and
    preprocessing configuration is suitable for training a full model.
    No final trainer, no artifact saving.
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
        model_trainer: IModelTrainer,
        evaluator: ICrossValidationEvaluator,
        visualizer: IVisualizer,
    ) -> None:
        """Initialize the evaluation pipeline with all required stage components."""
        self._log = logging.getLogger(__name__)
        self._data_loader = data_loader
        self._raw_preprocessing = raw_preprocessing
        self._raw_augmentation = raw_augmentation
        self._paradigm = paradigm
        self._epoch_preprocessing = epoch_preprocessing
        self._splitting = splitting
        self._augmentation = augmentation
        self._dataset_exporter = dataset_exporter
        self._model_trainer = model_trainer
        self._evaluator = evaluator
        self._visualizer = visualizer

    def run(self, config: ExperimentConfig, run_ctx: RunContext) -> None:
        """
        Execute the evaluation pipeline sequentially.

        Stages:
            1. Data loading
            2. Raw preprocessing + visualization
            3. Raw augmentation + visualization
            4. Paradigm extraction (epoching)
            5. Epoch preprocessing + visualization
            6. MOABB-based splitting into cross-validation folds (no validation_data required)
            7. Augmentation + visualization
            8. Dataset export
            9. Fold-based model training (small ML models only)
            10. Fold-level evaluation (each model evaluated on its fold's test data)

        Args:
            config: The master experiment configuration.
            run_ctx: Context keeping track of unique runtime metadata.

        Raises:
            ValueError: If splitting produces no folds.
        """
        # --- 1. Data Loading ---
        load_result: StepResult[RawDataDTO] = self._data_loader.run(config.source, run_ctx)

        # --- 2. Raw Preprocessing ---
        raw_preprocessing_input = RawPreprocessingInputDTO(config.raw_preprocessing, load_result.data)
        raw_preprocessing_result: StepResult[RawPreprocessedDTO] = self._raw_preprocessing.run(raw_preprocessing_input, run_ctx)
        self._visualizer.visualize_raw(raw_preprocessing_result.data, run_ctx)

        # --- 3. Raw Augmentation ---
        raw_augmentation_input = RawAugmentationInputDTO(config.raw_augmentation, raw_preprocessing_result.data)
        raw_augmentation_result = self._raw_augmentation.run(raw_augmentation_input, run_ctx)

        copies_per_sample_raw = getattr(config.raw_augmentation, "copies_per_sample", 0)
        self._visualizer.visualize_raw_augmentation(raw_augmentation_result.data, run_ctx, copies_per_sample_raw)

        # --- 4. Paradigm Extraction ---
        paradigm_input = ParadigmInputDTO(config.paradigm, raw_augmentation_result.data)
        paradigm_result: StepResult[ParadigmResultDTO] = self._paradigm.run(paradigm_input, run_ctx)

        # --- 5. Epoch Preprocessing ---
        epoch_preprocessing_input = EpochPreprocessingInputDTO(config.epoch_preprocessing, paradigm_result.data)
        epoch_preprocessing_result = self._epoch_preprocessing.run(epoch_preprocessing_input, run_ctx)
        self._visualizer.visualize_epochs(epoch_preprocessing_result.data, run_ctx)

        # --- 6. MOABB Splitting into folds ---
        splitting_input = SplitInputDTO(config.split, epoch_preprocessing_result.data)
        splitting_result: StepResult[DatasetSplitDTO] = self._splitting.run(splitting_input, run_ctx)

        # --- 7. Augmentation ---
        augmentation_input = AugmentationInputDTO(config.augmentation, splitting_result.data)
        augmentation_result = self._augmentation.run(augmentation_input, run_ctx)

        copies_per_sample = getattr(config.augmentation, "copies_per_sample", 0)
        self._visualizer.visualize_augmentation(augmentation_result.data, run_ctx, copies_per_sample)

        # --- 8. Dataset Export ---
        self._dataset_exporter.run(config.dataset_export, augmentation_result.data, run_ctx)

        # --- 9. Fold-based Model Training ---
        folds = augmentation_result.data.folds
        if not folds:
            raise ValueError("Splitting/Augmentation returned no folds. Cannot continue evaluation.")

        self._log.info(f"Training {len(folds)} fold model(s) for evaluation.")
        training_input = TrainingInputDTO(config=config.model, folds=folds)
        model_training_result: StepResult[TrainingResultDTO] = self._model_trainer.run(training_input, run_ctx)

        # --- 10. Fold-level Evaluation ---
        self._log.info("Evaluating fold-trained models on their respective fold test data.")
        fold_evaluation_input = CrossValidationEvaluationInputDTO(
            config=config.evaluation,
            trained_models=model_training_result.data.trained_models,
            folds=folds,
        )
        self._evaluator.run(fold_evaluation_input, run_ctx)

        self._log.info("Evaluation pipeline finished successfully.")
