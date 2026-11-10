"""Inference evaluator for the experiment pipeline."""

import numpy as np
from sklearn.metrics import confusion_matrix, get_scorer

from src.pipeline.context.run_context import RunContext
from src.pipeline.contracts.step_result import StepResult
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_input_dto import ExperimentEvaluationInputDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.interfaces.evaluator import IExperimentEvaluator


class ExperimentEvaluator(IExperimentEvaluator):
    """
    Inference evaluator for the experiment pipeline.

    Applies a pre-trained model to all supplied data.  Labels are optional:

    - **Labels present** → computes predictions, metrics, and confusion matrix.
    - **Labels absent** → returns predictions only (pure inference mode).
    """

    def run(
        self,
        input_dto: ExperimentEvaluationInputDTO,
        run_ctx: RunContext,
    ) -> StepResult[EvaluationResultDTO]:
        """
        Run inference on all data and optionally compute metrics.

        Stages:
            1. Initialization: validate that data is present.
            2. Data Extraction: build X (and optionally y) from ``data``.
            3. Inference: generate predictions (and optionally probabilities).
            4. Metrics (optional): if labels are available compute metrics and
               confusion matrix; otherwise return predictions-only result.

        Args:
            input_dto: DTO with a loaded model, all available data, and config.
            run_ctx: Runtime context (logging, run ID, …).

        Returns:
            ``EvaluationResultDTO`` with predictions always populated; metrics
            and targets populated only when labels are available in the data.

        Raises:
            ValueError: If ``data`` is missing or empty.
        """
        log = run_ctx.logger.for_step("EXPERIMENT_EVALUATOR")

        # --- 1. Initialization ---
        if not input_dto.data or not input_dto.data.data:
            log.error("No data found. ExperimentEvaluator requires non-empty 'data'.")
            raise ValueError("No data found. ExperimentEvaluator requires non-empty 'data'.")

        log.info(
            f"Starting inference for model '{input_dto.trained_model.model_name}'.",
            VerbosityLevel.QUIET,
        )

        # --- 2. Data Extraction ---
        x, y_true_or_none = self._extract_data(input_dto.data)
        has_labels = y_true_or_none is not None and len(y_true_or_none) > 0

        if has_labels:
            log.info("Labels detected — metrics will be computed.", VerbosityLevel.NORMAL)
        else:
            log.info("No labels detected — running in pure inference mode (predictions only).", VerbosityLevel.NORMAL)

        # --- 3. Inference ---
        log.info("Running model inference…", VerbosityLevel.DETAILED)
        y_pred = input_dto.trained_model.model.predict(x)

        probabilities = None
        try:
            probabilities = input_dto.trained_model.model.predict_class_probability(x)
        except (AttributeError, NotImplementedError):
            pass

        # --- 4. Metrics (optional) ---
        if not has_labels:
            result = EvaluationResultDTO(
                metrics={},
                predictions=y_pred.tolist(),
                targets=None,
                probabilities=probabilities.tolist() if probabilities is not None else None,
                confusion_matrix=None,
            )
            log.info("Inference completed (no labels — no metrics).", VerbosityLevel.QUIET)
            return StepResult(result)

        y_true: np.ndarray = y_true_or_none  # type: ignore[assignment]

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

        log.info("Inference and cross_validation completed successfully.", VerbosityLevel.QUIET)
        return StepResult(result)

    def _extract_data(self, preprocessed_data: EpochPreprocessedDTO) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Extract X and optionally y from an ``EpochPreprocessedDTO``.

        Returns:
            Tuple of (X, y) where y is ``None`` when no labels are found in
            any recording.
        """
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        has_any_labels = False

        for recording in preprocessed_data.data:
            epochs = recording.data
            if hasattr(epochs, "get_data"):
                x_list.append(epochs.get_data(copy=False))
                labels = epochs.events[:, -1]
                y_list.append(labels)
                has_any_labels = True
            else:
                x_list.append(epochs)
                raw_labels = recording.metadata.get("labels", [])
                if len(raw_labels) > 0:
                    y_list.append(np.array(raw_labels))
                    has_any_labels = True
                else:
                    # Append empty placeholder so indices stay aligned
                    y_list.append(np.array([], dtype=np.int64))

        x = np.concatenate(x_list, axis=0)
        if not has_any_labels:
            return x, None
        y = np.concatenate(y_list, axis=0)
        return x, y if len(y) > 0 else None
