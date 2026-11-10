import numpy as np
from sklearn.metrics import confusion_matrix, get_scorer

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_input_dto import TrainingEvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import ITrainingEvaluator


class TrainingEvaluator(ITrainingEvaluator):
    """
    Evaluates a single final model on a held-out test set (TrainingPipeline).

    Receives one trained model and a dedicated ``test_data`` partition that
    the model has never seen during training, computes predictions and the
    requested metrics, and returns the result.
    """

    def run(
        self,
        input_dto: TrainingEvaluationInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[EvaluationResultDTO]:
        """
        Evaluate the model on the held-out test partition.

        Stages:
            1. Initialization: validate that a model and test data are present.
            2. Data Extraction: build (X, y) arrays from ``test_data``.
            3. Model Evaluation: generate predictions (and optionally probabilities).
            4. Metrics Computation: compute all configured metrics and confusion matrix.

        Args:
            input_dto: DTO with one trained model, test data, and cross_validation config.
            run_ctx: Runtime context (logging, run ID, …).

        Returns:
            ``EvaluationResultDTO`` with metrics computed on the test partition.

        Raises:
            ValueError: If test data is missing or empty.
        """
        log = run_ctx.logger.for_step("TRAINING_EVALUATION")

        # --- 1. Initialization ---
        if not input_dto.test_data or not input_dto.test_data.data:
            log.error("No test data found. TrainingEvaluator requires a non-empty 'test_data'.")
            raise ValueError("No test data found. TrainingEvaluator requires a non-empty 'test_data'.")

        log.info(
            f"Starting cross_validation for model '{input_dto.trained_model.model_name}' on test data.",
            VerbosityLevel.QUIET,
        )

        # --- 2. Data Extraction ---
        x_test, y_true = self._extract_data(input_dto.test_data)

        # --- 3. Model Evaluation ---
        log.info("Evaluating model…", VerbosityLevel.DETAILED)
        y_pred = input_dto.trained_model.model.predict(x_test)

        probabilities = None
        try:
            probabilities = input_dto.trained_model.model.predict_class_probability(x_test)
        except (AttributeError, NotImplementedError):
            pass

        # --- 4. Metrics Computation ---
        model_metrics: dict[str, float] = {}
        for m_name in input_dto.config.metrics:
            scorer = get_scorer(m_name)
            if hasattr(scorer, "_score_func"):
                val = float(scorer._score_func(y_true, y_pred, **scorer._kwargs))
                model_metrics[m_name] = val
                log.info(f"Metric {m_name}: {val:.4f}", VerbosityLevel.TRACE)
            else:
                log.warning(f"Could not calculate metric '{m_name}' directly from labels.")

        overall_cm = confusion_matrix(y_true, y_pred).tolist()

        result = EvaluationResultDTO(
            metrics=model_metrics,
            predictions=y_pred.tolist(),
            targets=y_true.tolist(),
            probabilities=probabilities.tolist() if probabilities is not None else None,
            confusion_matrix=overall_cm,
        )

        log.info("Evaluation completed successfully.", VerbosityLevel.QUIET)
        return StepResult(result)

    def _extract_data(self, preprocessed_data: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray]:
        """Extract (X, y) arrays from an ``EpochPreprocessedDTO``."""
        x_list, y_list = [], []
        for recording in preprocessed_data.data:
            epochs = recording.data
            if hasattr(epochs, "get_data"):
                x_list.append(epochs.get_data(copy=False))
                y_list.append(epochs.events[:, -1])
            else:
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)
