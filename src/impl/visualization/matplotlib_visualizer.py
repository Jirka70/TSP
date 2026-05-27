import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.visualization_config import VisualizationConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.raw_augmentation.raw_augmented_dto import RawAugmentedDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.visualizer import IVisualizer
from src.types.dto.config.logging.verbosity_level import VerbosityLevel


class MatplotlibVisualizer(IVisualizer):
    """
    Implementation of IVisualizer using Matplotlib and Seaborn for static image reports.

    This visualizer generates high-quality static plots (PNG) for different stages of the
    EEG processing pipeline, including PSD plots, ERP averages, and comprehensive
    evaluation reports with confusion matrices and class distributions.
    """

    def __init__(self, config: VisualizationConfig) -> None:
        """
        Initializes the visualizer with configuration.

        Args:
            config (VisualizationConfig): Configuration for visualization settings (width, height, n_fft, etc.).
        """
        self._config = config

        # --- 1. Dimensions Setup ---
        # We assume config.width and config.height are in PIXELS and convert them to inches for Matplotlib.
        self._dpi = 100
        self._fig_width = config.width / self._dpi
        self._fig_height = config.height / self._dpi

    def visualize_raw(self, data: RawPreprocessedDTO, run_ctx: RunContext) -> None:
        """
        Visualizes Power Spectral Density (PSD) of the first recording to check preprocessing quality.

        The process follows these stages:
        1. Initialization: Checks configuration and determines safe n_fft for frequency analysis.
        2. PSD Plotting: Utilizes MNE's native plotting tools or fallbacks to NumPy traces.
        3. Formatting: Applies standardized labels, titles, and figure dimensions.
        4. Output: Saves the static plot and optionally displays it.

        Args:
            data (RawPreprocessedDTO): Contains preprocessed raw EEG recordings.
            run_ctx (RunContext): Pipeline execution context.
        """
        if not self._config.visualize_raw or not data.data:
            return

        log = run_ctx.logger.for_step("MATPLOTLIB_VISUALISATION_RAW")
        log.info("Visualizing PSD of preprocessed raw data.", VerbosityLevel.QUIET)

        # --- 1. Initialization ---
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

        # --- 2. PSD Plotting ---
        # Suppress MNE/Scipy warnings about nperseg > length
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*nperseg.*")

            if hasattr(raw, "compute_psd"):
                # MNE Raw object
                try:
                    # Use adjusted n_fft and explicitly set n_per_seg to avoid warnings
                    psd = raw.compute_psd(fmax=50, n_fft=n_fft, n_per_seg=n_fft, verbose=False)
                    fig = psd.plot(show=False)
                    # --- 3. Formatting ---
                    fig.set_size_inches(self._fig_width, self._fig_height)
                    fig.set_dpi(self._dpi)
                    fig.suptitle(f"PSD: Subject {recording.subject_id}")
                except Exception as e:
                    log.warning(f"Could not plot PSD: {e}")
                    return
            elif hasattr(raw, "plot_psd"):
                # Older MNE versions
                fig = raw.plot_psd(show=False, fmax=50, n_fft=n_fft, n_per_seg=n_fft, verbose=False)
                # --- 3. Formatting ---
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

        # --- 4. Output ---
        self._handle_output("raw_preprocessing_psd.png", run_ctx)

    def visualize_raw_augmentation(self, data: RawAugmentedDTO, run_ctx: RunContext, copies_per_sample: int = 0) -> None:
        """
        Visualizes raw augmented data comparison using a multi-subplot layout.

        Displays the original signal and its augmented variations side-by-side:
        1. Signal Selection: Identifies the first original recording and its augmented copies.
        2. Layout Setup: Initializes a multi-row figure based on the number of copies.
        3. Subplot Generation: Processes each recording (MNE or NumPy) and renders its trace.
        4. Finishing: Applies tight layout and saves the comparison plot.

        Args:
            data (RawAugmentedDTO): Contains original and augmented raw recordings.
            run_ctx (RunContext): Pipeline execution context.
            copies_per_sample (int): Number of augmented copies per sample.
        """
        if not self._config.visualize_raw_augmentation or not data.data:
            return

        log = run_ctx.logger.for_step("MATPLOTLIB_VISUALISATION_RAW_AUGMENTED")
        log.info("Visualizing raw augmented data.", VerbosityLevel.QUIET)

        # --- 1. Signal Selection ---
        n_copies = 1 + copies_per_sample
        recordings = data.data[:n_copies]

        # --- 2. Layout Setup ---
        plt.figure(figsize=(self._fig_width, self._fig_height), dpi=self._dpi)

        # --- 3. Subplot Generation ---
        for i, recording in enumerate(recordings):
            plt.subplot(n_copies, 1, i + 1)
            raw = recording.data
            # Limit samples for clarity if signal is too long
            max_samples = 2000
            n_times = int(raw.n_times) if hasattr(raw, "n_times") else 0
            stop_idx = min(max_samples, n_times) if n_times > 0 else max_samples

            if hasattr(raw, "get_data"):
                ch_data = raw.get_data(picks=[0], stop=stop_idx)
                times = raw.times[: ch_data.shape[1]]
                ch_name = raw.ch_names[0]
            else:
                # Fallback for NumPy
                ch_data = raw[0, :stop_idx] if isinstance(raw, np.ndarray) else np.array([])
                times = np.arange(ch_data.shape[0])
                ch_name = "0"

            plt.plot(times, ch_data[0, :] if ch_data.ndim > 1 else ch_data)
            title = "Original Signal" if i == 0 else f"Augmented Copy {i}"
            plt.title(f"{title} - Subject {recording.subject_id}, Channel {ch_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")

        # --- 4. Finishing ---
        plt.tight_layout()
        self._handle_output("raw_augmentation_comparison.png", run_ctx)

    def visualize_epochs(self, data: EpochPreprocessedDTO, run_ctx: RunContext) -> None:
        """
        Visualizes Event-Related Potential (ERP) averages of the epoched data.

        The process follows these steps:
        1. ERP Calculation: Averages epochs to isolate the underlying neural response.
        2. Plotting Strategy: Uses MNE's `average().plot()` or falls back to NumPy mean traces.
        3. Formatting: Adjusts labels and titles for publication-ready static images.
        4. Output: Finalizes the figure and saves to the output directory.

        Args:
            data (EpochPreprocessedDTO): Preprocessed epoched data.
            run_ctx (RunContext): Pipeline execution context.
        """
        if not self._config.visualize_epochs or not data.data:
            return

        log = run_ctx.logger.for_step("MATPLOTLIB_VISUALISATION_EPOCHS")
        log.info("Visualizing ERP of epoched data.", VerbosityLevel.QUIET)

        # --- 1. ERP Calculation & 2. Plotting Strategy ---
        recording = data.data[0]
        epochs = recording.data

        if hasattr(epochs, "average"):
            # MNE Epochs object
            evoked = epochs.average()
            fig = evoked.plot(show=False)
            # --- 3. Formatting ---
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

        # --- 4. Output ---
        self._handle_output("epoching_erp.png", run_ctx)

    def visualize_augmentation(self, data: DatasetSplitDTO, run_ctx: RunContext, copies_per_sample: int = 0) -> None:
        """
        Visualizes augmented data comparison for the first fold.

        Facilitates variety checks for augmentation strategies applied to epoched data:
        1. Fold Selection: Focuses on the first training partition.
        2. Index Mapping: Maps original sample indices to their augmented counterparts.
        3. Trace Comparison: Renders a vertical stack of traces for the original and its copies.
        4. Finishing: Applies styling and saves the fold-specific comparison plot.

        Args:
            data (DatasetSplitDTO): Dataset containing generated folds and augmented samples.
            run_ctx (RunContext): Pipeline execution context.
            copies_per_sample (int): Number of augmented copies per sample.
        """
        if not self._config.visualize_augmentation or not data.folds:
            return

        log = run_ctx.logger.for_step("MATPLOTLIB_VISUALISATION_AUGMENTED")
        log.info("Visualizing augmented data.", VerbosityLevel.QUIET)

        # --- 1. Fold Selection ---
        fold = data.folds[0]
        if not fold.train_data or not fold.train_data.data:
            return

        recording = fold.train_data.data[0]
        x = recording.data

        # --- 2. Index Mapping ---
        # If it's augmented, it should be a NumPy array now
        if isinstance(x, np.ndarray) and x.ndim == 3:
            plt.figure(figsize=(self._fig_width, self._fig_height), dpi=self._dpi)

            n_original_samples = x.shape[0] // (1 + copies_per_sample)
            # We want to show the first original sample and its augmented copies
            indices_to_plot = [0] + [(i + 1) * n_original_samples for i in range(copies_per_sample)]

            # --- 3. Trace Comparison ---
            for i, idx in enumerate(indices_to_plot):
                plt.subplot(len(indices_to_plot), 1, i + 1)
                plt.plot(x[idx, 0, :])  # First channel
                title = "Original Sample" if i == 0 else f"Augmented Copy {i}"
                plt.title(f"{title} (Fold {fold.fold_idx}, Subject {recording.subject_id})")

            # --- 4. Finishing ---
            plt.tight_layout()
            self._handle_output(f"augmentation_fold_{fold.fold_idx}.png", run_ctx)

    def visualize_evaluation(self, data: EvaluationResultDTO, run_ctx: RunContext, model_name: str) -> None:
        """
        Visualizes evaluation results including Confusion Matrix, Class Distribution, and Metrics.

        Generates a multi-panel report for comprehensive model analysis:
        1. Confusion Matrix: Heatmap showing classification hits and misses.
        2. Class Distribution: Bar chart comparing reality vs. model predictions.
        3. Metrics Summary: Horizontal bar chart for aggregate performance scores.

        Args:
            data (EvaluationResultDTO): Contains classification metrics and confusion matrix.
            run_ctx (RunContext): Pipeline execution context.
            model_name (str): Name of the model being evaluated.
        """
        if not self._config.visualize_evaluation:
            return

        log = run_ctx.logger.for_step("MATPLOTLIB_VISUALISATION_EVALUATION")
        log.info("Visualizing evaluation results.", VerbosityLevel.QUIET)

        if not data.targets or not data.predictions or data.confusion_matrix is None:
            log.warning("Insufficient data for evaluation visualization.")
            return

        plt.figure(figsize=(self._fig_width, self._fig_height), dpi=self._dpi)

        # --- 1. Confusion Matrix ---
        plt.subplot(1, 3, 1)
        sns.heatmap(data.confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")

        # --- 2. Class Distribution ---
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

        # --- 3. Metrics Summary ---
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
        self._handle_output(f"evaluation_{model_name.lower().replace(' ', '_')}.png", run_ctx)

    def _handle_output(self, filename: str, run_ctx: RunContext) -> None:
        """Handles saving and showing of the current plot."""
        log = run_ctx.logger.for_step("SAVE_PLOTS")
        if self._config.save_plots:
            output_dir = run_ctx.output_dir
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            save_path = plots_dir / filename
            plt.savefig(str(save_path), dpi=self._dpi)
            log.info(f"Plot saved to: {save_path}", VerbosityLevel.DETAILED)

        if self._config.show_plots:
            plt.show()  # This blocks until the window is closed!

        plt.close()  # Always close to free memory and prevent blocking
