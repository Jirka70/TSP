import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.core.hydra_config import HydraConfig

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.visualization_config import VisualizationConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.visualizer import IVisualizer

log = logging.getLogger(__name__)


class MatplotlibVisualizer(IVisualizer):
    """Implementation of IVisualizer using Matplotlib and Seaborn."""

    def __init__(self, config: VisualizationConfig) -> None:
        """Initializes the visualizer with configuration.

        Args:
            config (VisualizationConfig): Configuration for visualization.
        """
        self._config = config

        # We assume config.width and config.height are in PIXELS and convert them to inches for Matplotlib.
        self._dpi = 100
        self._fig_width = config.width / self._dpi
        self._fig_height = config.height / self._dpi

    def visualize_raw(self, data: RawPreprocessedDTO, run_ctx: RunContext) -> None:
        """Visualizes PSD of the first recording to check preprocessing quality."""
        if not self._config.visualize_raw or not data.data:
            return

        log.info("Visualizing PSD of preprocessed raw data...")
        recording = data.data[0]
        raw = recording.data

        # Determine safe n_fft (must be <= signal length and ideally a power of 2)
        n_times = int(raw.n_times) if hasattr(raw, "n_times") else 0
        target_n_fft = self._config.n_fft

        if n_times > 0:
            # Ensure n_fft is a power of 2 and <= n_times
            n_fft = 1 << (min(n_times, target_n_fft).bit_length() - 1)
        else:
            n_fft = target_n_fft

        # Suppress MNE/Scipy warnings about nperseg > length

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*nperseg.*")

            if hasattr(raw, "compute_psd"):
                # MNE Raw object
                try:
                    # Use adjusted n_fft and explicitly set n_per_seg to avoid warnings
                    psd = raw.compute_psd(fmax=50, n_fft=n_fft, n_per_seg=n_fft, verbose=False)
                    fig = psd.plot(show=False)
                    fig.set_size_inches(self._fig_width, self._fig_height)
                    fig.set_dpi(self._dpi)
                    fig.suptitle(f"PSD: Subject {recording.subject_id}")
                except Exception as e:
                    log.warning(f"Could not plot PSD: {e}")
                    return
            elif hasattr(raw, "plot_psd"):
                # Older MNE versions
                fig = raw.plot_psd(show=False, fmax=50, n_fft=n_fft, n_per_seg=n_fft, verbose=False)
                fig.set_size_inches(self._fig_width, self._fig_height)
                fig.set_dpi(self._dpi)
                fig.suptitle(f"PSD: Subject {recording.subject_id}")
            else:
                # Fallback for NumPy
                plt.figure(figsize=(self._fig_width, self._fig_height), dpi=self._dpi)
                # Assuming (channels, time)
                if isinstance(raw, np.ndarray) and raw.ndim >= 2:
                    plt.plot(raw[0, : min(1000, raw.shape[1])])
                plt.title(f"Raw Signal Trace - Subject {recording.subject_id}")

        self._handle_output("raw_preprocessing_psd.png")

    def visualize_epochs(self, data: EpochPreprocessedDTO, run_ctx: RunContext) -> None:
        """Visualizes ERP (average) of the epoched data."""
        if not self._config.visualize_epochs or not data.data:
            return

        log.info("Visualizing ERP of epoched data...")
        recording = data.data[0]
        epochs = recording.data

        if hasattr(epochs, "average"):
            # MNE Epochs object
            evoked = epochs.average()
            fig = evoked.plot(show=False)
            fig.set_size_inches(self._fig_width, self._fig_height)
            fig.set_dpi(self._dpi)
            fig.suptitle(f"ERP Average: Subject {recording.subject_id}")
        else:
            # Fallback for NumPy (mean across epochs)
            plt.figure(figsize=(self._fig_width, self._fig_height), dpi=self._dpi)
            if isinstance(epochs, np.ndarray) and epochs.ndim == 3:
                erp = np.mean(epochs, axis=0)
                plt.plot(erp[0])  # Plot average of first channel
            plt.title(f"ERP Average - Subject {recording.subject_id}")

        self._handle_output("epoching_erp.png")

    def visualize_augmentation(self, data: DatasetSplitDTO, run_ctx: RunContext) -> None:
        """Visualizes augmented data comparison for the first fold."""
        if not self._config.visualize_augmentation or not data.folds:
            return

        log.info("Visualizing augmented data...")
        # Take first fold and first recording for visualization
        fold = data.folds[0]
        if not fold.train_data or not fold.train_data.data:
            return

        recording = fold.train_data.data[0]
        x = recording.data

        # If it's augmented, it should be a NumPy array now
        if isinstance(x, np.ndarray) and x.ndim == 3:
            plt.figure(figsize=(self._fig_width, self._fig_height), dpi=self._dpi)

            n_samples = x.shape[0]
            for i in range(n_samples):
                plt.subplot(n_samples, 1, i + 1)
                plt.plot(x[i, 0, :])  # First channel
                plt.title(f"Sample {i} (Fold {fold.fold_idx}, Subject {recording.subject_id})")

            plt.tight_layout()
            self._handle_output(f"augmentation_fold_{fold.fold_idx}.png")

    def visualize_evaluation(self, data: EvaluationResultDTO, run_ctx: RunContext, model_name: str) -> None:
        """Visualizes evaluation results (Confusion Matrix and Metrics)."""
        if not self._config.visualize_evaluation:
            return

        if not data.targets or not data.predictions or data.confusion_matrix is None:
            log.warning("Insufficient data for evaluation visualization.")
            return

        plt.figure(figsize=(self._fig_width, self._fig_height), dpi=self._dpi)

        # 1. Confusion Matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(data.confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")

        # 2. Class distribution
        plt.subplot(1, 3, 2)
        y_true = np.array(data.targets)
        y_pred = np.array(data.predictions)
        classes, counts_true = np.unique(y_true, return_counts=True)
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        pred_counts_dict = dict(zip(pred_unique, pred_counts, strict=True))
        counts_pred = [pred_counts_dict.get(cls, 0) for cls in classes]

        x = np.arange(len(classes))
        width = 0.35
        plt.bar(x - width / 2, counts_true, width, label="Reality", color="gray", alpha=0.6)
        plt.bar(x + width / 2, counts_pred, width, label="Predicted", color="skyblue")

        plt.title("Class distribution")
        plt.xticks(x, classes)
        plt.legend()

        # 3. Metrics Summary
        plt.subplot(1, 3, 3)
        m_names = list(data.metrics.keys())
        m_values = [data.metrics[name] for name in m_names]

        bars = plt.barh(m_names, m_values, color="salmon")
        plt.xlim(0, 1.1)
        plt.title("Aggregate Metrics")

        for bar in bars:
            val = bar.get_width()
            plt.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontweight="bold")

        plt.tight_layout()
        logging.info("Saving plot to file...")
        self._handle_output(f"evaluation_{model_name.lower().replace(' ', '_')}.png")

    def _handle_output(self, filename: str) -> None:
        """Handles saving and showing of the current plot."""
        if self._config.save_plots:
            try:
                output_dir = Path(HydraConfig.get().runtime.output_dir).absolute()
            except (ValueError, KeyError, RuntimeError):
                output_dir = Path(os.getcwd()).absolute()

            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            save_path = plots_dir / filename
            plt.savefig(str(save_path), dpi=self._dpi)
            log.info(f"Plot saved to: {save_path}")

        if self._config.show_plots:
            plt.show()  # This blocks until the window is closed!

        plt.close()  # Always close to free memory and prevent blocking
