"""Fold-level evaluator for the evaluation pipeline."""

import numpy as np
from sklearn.metrics import confusion_matrix, get_scorer

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import IEvaluator


class FoldEvaluator(IEvaluator):
    """
    A fold-level evaluator designed for the evaluation pipeline.

    Unlike StandardEvaluator (which evaluates on global validation data),
    FoldEvaluator evaluates each trained model on its corresponding fold's
    test data (fold.test_data). This is appropriate for cross-validation
    scenarios where no separate global validation set is produced.

    The evaluator iterates over all trained models, evaluates each on its
    fold's test set, and aggregates the per-fold metrics by computing the
    mean across folds.
    """

    def run(self, input_dto: EvaluationInputDTO, run_ctx: RunContext) -> StepResult[EvaluationResultDTO]:
        """
        Evaluates each trained model on the test data of its respective fold.

        The process follows these stages:
        1. Initialization: Validates that models and fold test data are available.
        2. Per-fold Evaluation: For each trained model, extracts test data from the
           corresponding fold and computes predictions and metrics.
        3. Aggregation: Averages per-fold metrics across all folds and concatenates
           predictions and targets for a combined confusion matrix.

        Args:
            input_dto (EvaluationInputDTO): DTO containing trained models and dataset split.
                The dataset_split must contain folds with test_data populated.
            run_ctx (RunContext): Context keeping track of the current pipeline execution.

        Returns:
            StepResult[EvaluationResultDTO]: Aggregated evaluation results across all folds.

        Raises:
            ValueError: If no models are provided, if dataset_split is missing, or if
                fold test data is unavailable for a given fold.
        """
        log = run_ctx.logger.for_step("FOLD_EVALUATOR")

        # --- 1. Initialization ---
        if not input_dto.trained_models:
            log.info("EvaluationInputDTO does not include any model. Skipping evaluation.", VerbosityLevel.NORMAL)
            return StepResult(EvaluationResultDTO(
                metrics={},
                predictions=[],
                targets=[],
                probabilities=None,
                confusion_matrix=None,
            ))

        if input_dto.dataset_split is None or not input_dto.dataset_split.folds:
            log.error("FoldEvaluator requires dataset_split with folds. No folds found.")
            raise ValueError(
                "FoldEvaluator requires dataset_split with populated folds. "
                "Ensure the splitter produces folds with test_data."
            )

        folds = input_dto.dataset_split.folds
        trained_models = input_dto.trained_models

        if len(trained_models) != len(folds):
            log.error(
                f"Number of trained models ({len(trained_models)}) does not match "
                f"number of folds ({len(folds)}). Cannot perform fold-level evaluation."
            )
            raise ValueError(
                f"Mismatch between trained models ({len(trained_models)}) "
                f"and folds ({len(folds)})."
            )

        log.info(
            f"Starting fold-level evaluation for {len(trained_models)} model(s) across {len(folds)} fold(s).",
            VerbosityLevel.QUIET,
        )

        # --- 2. Per-fold Evaluation ---
        all_metrics: list[dict[str, float]] = []
        all_predictions: list[int] = []
        all_targets: list[int] = []
        all_probabilities: list[list[float]] = []

        for fold_dto, model_dto in zip(folds, trained_models):
            if fold_dto.test_data is None or not fold_dto.test_data.data:
                log.error(f"Fold {fold_dto.fold_idx} has no test_data. Cannot evaluate model.")
                raise ValueError(
                    f"Fold {fold_dto.fold_idx} is missing test_data. "
                    "Ensure the splitter produces test splits for all folds."
                )

            log.info(
                f"Evaluating fold {fold_dto.fold_idx} with model '{model_dto.model_name}'.",
                VerbosityLevel.NORMAL,
            )

            x_test, y_true = self._extract_data(fold_dto.test_data)

            # Predict labels
            y_pred = model_dto.model.predict(x_test)

            # Predict probabilities if supported
            fold_probabilities = None
            try:
                fold_probabilities = model_dto.model.predict_class_probability(x_test)
            except (AttributeError, NotImplementedError):
                pass

            # Compute metrics for this fold
            fold_metrics: dict[str, float] = {}
            for m_name in input_dto.config.metrics:
                scorer = get_scorer(m_name)
                if hasattr(scorer, "_score_func"):
                    val = float(scorer._score_func(y_true, y_pred, **scorer._kwargs))
                    fold_metrics[m_name] = val
                    log.info(f"  Fold {fold_dto.fold_idx} — {m_name}: {val:.4f}", VerbosityLevel.TRACE)
                else:
                    log.warning(f"Could not calculate metric '{m_name}' from fold {fold_dto.fold_idx}.")

            all_metrics.append(fold_metrics)
            all_predictions.extend(y_pred.tolist())
            all_targets.extend(y_true.tolist())
            if fold_probabilities is not None:
                all_probabilities.extend(fold_probabilities.tolist())

        # --- 3. Aggregation ---
        aggregated_metrics: dict[str, float] = {}
        for m_name in input_dto.config.metrics:
            per_fold_vals = [fm[m_name] for fm in all_metrics if m_name in fm]
            if per_fold_vals:
                mean_val = float(np.mean(per_fold_vals))
                aggregated_metrics[m_name] = mean_val
                log.info(f"Aggregated (mean) {m_name}: {mean_val:.4f}", VerbosityLevel.QUIET)

        combined_cm = confusion_matrix(all_targets, all_predictions).tolist() if all_targets else None

        result = EvaluationResultDTO(
            metrics=aggregated_metrics,
            predictions=all_predictions,
            targets=all_targets,
            probabilities=all_probabilities if all_probabilities else None,
            confusion_matrix=combined_cm,
            metadata={
                "n_folds": len(folds),
                "per_fold_metrics": all_metrics,
            },
        )

        log.info("Fold-level evaluation completed successfully.", VerbosityLevel.QUIET)
        return StepResult(result)

    def _extract_data(self, preprocessed_data: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts and concatenates the feature matrix (X) and labels (y) from a preprocessed dataset.

        Args:
            preprocessed_data (EpochPreprocessedDTO): The data transfer object containing
                a list of preprocessed recordings.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple of (X, y) arrays.
        """
        x_list = []
        y_list = []

        for recording in preprocessed_data.data:
            epochs = recording.data
            if hasattr(epochs, "get_data"):
                # Handle MNE Epochs
                x_list.append(epochs.get_data(copy=False))
                y_list.append(epochs.events[:, -1])
            else:
                # Handle NumPy arrays
                x_list.append(epochs)
                y_list.append(np.array(recording.metadata.get("labels", [])))

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return x, y
