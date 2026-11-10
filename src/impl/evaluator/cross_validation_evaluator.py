"""Fold-level evaluator for the cross_validation pipeline."""

import numpy as np
from sklearn.metrics import confusion_matrix, get_scorer

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_input_dto import CrossValidationEvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import ICrossValidationEvaluator


class CrossValidationEvaluator(ICrossValidationEvaluator):
    """
    Fold-level evaluator for the cross_validation pipeline.

    Pairs each trained model with its corresponding fold by ``fold_idx``,
    evaluates the model on ``fold.test_data``, and aggregates per-fold
    metrics into a single result (mean across folds, combined confusion matrix).
    """

    def run(
        self,
        input_dto: CrossValidationEvaluationInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[EvaluationResultDTO]:
        """
        Evaluate each model on its fold's test data and return aggregated metrics.

        Stages:
            1. Initialization: validate inputs and build a fold_idx → fold mapping.
            2. Per-fold Evaluation: predict and compute metrics for each model/fold pair.
            3. Aggregation: average per-fold metrics; build a combined confusion matrix.

        Args:
            input_dto: DTO with trained models, folds, and cross_validation config.
            run_ctx: Runtime context (logging, run ID, …).

        Returns:
            Aggregated ``EvaluationResultDTO`` with mean metrics across all folds.

        Raises:
            ValueError: If no models are provided, folds are missing, counts do not
                match, or a fold has no ``test_data``.
        """
        log = run_ctx.logger.for_step("CROSS_VALIDATION_EVALUATOR")

        # --- 1. Initialization ---
        if not input_dto.trained_models:
            log.info("No trained models provided. Skipping cross cross_validation.", VerbosityLevel.NORMAL)
            return StepResult(EvaluationResultDTO(metrics={}, predictions=[], targets=[], probabilities=None, confusion_matrix=None))

        if not input_dto.folds:
            log.error("CrossEvaluator requires at least one fold. No folds found.")
            raise ValueError("CrossEvaluator requires folds with test_data populated.")

        if len(input_dto.trained_models) != len(input_dto.folds):
            raise ValueError(f"Mismatch: {len(input_dto.trained_models)} trained model(s) vs {len(input_dto.folds)} fold(s). Counts must be equal.")

        # Build fold_idx → fold lookup so order in both lists does not matter.
        fold_by_idx = {fold.fold_idx: fold for fold in input_dto.folds}

        log.info(
            f"Starting fold-level cross_validation: {len(input_dto.trained_models)} model(s) / {len(input_dto.folds)} fold(s).",
            VerbosityLevel.QUIET,
        )

        # --- 2. Per-fold Evaluation ---
        all_metrics: list[dict[str, float]] = []
        all_predictions: list[int] = []
        all_targets: list[int] = []
        all_probabilities: list[list[float]] = []

        for model_dto in input_dto.trained_models:
            fold = fold_by_idx.get(model_dto.fold_idx)
            if fold is None:
                raise ValueError(f"No fold with fold_idx={model_dto.fold_idx} found in input folds. Available fold indices: {list(fold_by_idx.keys())}.")
            if fold.test_data is None or not fold.test_data.data:
                raise ValueError(f"Fold {fold.fold_idx} has no test_data. Ensure the splitter produces test splits for every fold.")

            log.info(f"Evaluating model '{model_dto.model_name}' on fold {fold.fold_idx}.", VerbosityLevel.NORMAL)

            x_test, y_true = self._extract_data(fold.test_data)
            y_pred = model_dto.model.predict(x_test)

            fold_probabilities = None
            try:
                fold_probabilities = model_dto.model.predict_class_probability(x_test)
            except (AttributeError, NotImplementedError):
                pass

            fold_metrics: dict[str, float] = {}
            for m_name in input_dto.config.metrics:
                scorer = get_scorer(m_name)
                if hasattr(scorer, "_score_func"):
                    val = float(scorer._score_func(y_true, y_pred, **scorer._kwargs))
                    fold_metrics[m_name] = val
                    log.info(f"  Fold {fold.fold_idx} — {m_name}: {val:.4f}", VerbosityLevel.TRACE)
                else:
                    log.warning(f"Could not calculate metric '{m_name}' for fold {fold.fold_idx}.")

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
                "n_folds": len(input_dto.folds),
                "per_fold_metrics": all_metrics,
            },
        )

        log.info("Fold-level cross_validation completed successfully.", VerbosityLevel.QUIET)
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
