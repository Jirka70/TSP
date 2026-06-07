import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.visualization_config import VisualizationConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.raw_augmentation.raw_augmented_dto import RawAugmentedDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.visualizer import IVisualizer
from src.types.dto.config.logging.verbosity_level import VerbosityLevel


class PlotlyVisualizer(IVisualizer):
    """
    Implementation of IVisualizer using Plotly for interactive HTML reports.

    This visualizer creates rich, interactive visualizations for different stages of the
    EEG processing pipeline, including raw PSD plots, ERP averages, and evaluation
    metrics (confusion matrices, etc.).
    """

    # --- Constants for automatic sizing ---
    DEFAULT_ROW_HEIGHT = 350
    AUGMENTATION_ROW_HEIGHT = 250

    def __init__(self, config: VisualizationConfig) -> None:
        """
        Initializes the visualizer with configuration.

        Args:
            config (VisualizationConfig): Configuration for visualization settings (n_fft, etc.).
        """
        self._config = config

    def visualize_raw(self, data: RawPreprocessedDTO, run_ctx: RunContext) -> None:
        """
        Visualizes Power Spectral Density (PSD) of the first recording using Plotly.

        The process follows these stages:
        1. Initialization: Checks configuration and determines safe n_fft for frequency analysis.
        2. PSD Computation: Uses MNE's compute_psd to calculate power across frequencies.
        3. Trace Preparation: Aggregates data across channels and prepares the Plotly trace.
        4. Output: Renders the interactive figure and saves it as an HTML file.

        Args:
            data (RawPreprocessedDTO): Contains preprocessed raw EEG recordings.
            run_ctx (RunContext): Pipeline execution context.
        """
        if not self._config.visualize_raw or not data.data:
            return

        log = run_ctx.logger.for_step("PLOTLY_VISUALISATION_RAW")
        log.info("Visualizing PSD (interactive) of preprocessed raw data...", VerbosityLevel.QUIET)

        # --- 1. Initialization ---
        recording = data.data[0]
        raw = recording.data

        # Determine safe n_fft based on signal length
        n_times = int(raw.n_times) if hasattr(raw, "n_times") else 0
        target_n_fft = self._config.n_fft

        if n_times > 0:
            # Ensure n_fft is a power of 2 and <= n_times
            n_fft = 1 << (min(n_times, target_n_fft).bit_length() - 1)
        else:
            n_fft = target_n_fft

        # --- 2. PSD Computation ---
        if hasattr(raw, "compute_psd"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*nperseg.*")
                psd = raw.compute_psd(fmax=50, n_fft=n_fft, n_per_seg=n_fft, verbose=False)
                data_arr, freqs = psd.get_data(return_freqs=True)

            # --- 3. Trace Preparation ---
            # Take average across channels for simplicity in the main plot
            psd_mean = np.mean(data_arr, axis=0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs, y=psd_mean, mode="lines", name="Mean PSD"))
            fig.update_layout(title=f"Power Spectral Density - Subject {recording.subject_id}", xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", template="plotly_white")

            # --- 4. Output ---
            self._handle_output(fig, "raw_psd_interactive.html", run_ctx)

    def visualize_raw_augmentation(self, data: RawAugmentedDTO, run_ctx: RunContext, copies_per_sample: int = 0) -> None:
        """
        Visualizes raw augmented data comparison using interactive Plotly subplots.

        This method displays a side-by-side comparison of the original signal and its
        augmented counterparts following these steps:
        1. Setup: Identifies original and augmented recordings.
        2. Subplot Generation: Creates a vertical stack of plots for each signal copy.
        3. Signal Preparation: Extracts and truncates time-series data for display.
        4. Plotting: Adds interactive traces to each subplot.

        Args:
            data (RawAugmentedDTO): Contains original and augmented raw recordings.
            run_ctx (RunContext): Pipeline execution context.
            copies_per_sample (int): Number of augmented copies generated per original sample.
        """
        if not self._config.visualize_raw_augmentation or not data.data:
            return

        log = run_ctx.logger.for_step("PLOTLY_VISUALISATION_RAW_AUGMENTED")
        log.info("Visualizing raw augmented samples (interactive)...", VerbosityLevel.QUIET)

        # --- 1. Setup ---
        n_copies = 1 + copies_per_sample
        recordings = data.data[:n_copies]

        # --- 2. Subplot Generation ---
        titles = ["Original Signal"] + [f"Augmented Copy {i}" for i in range(1, n_copies)]
        fig = make_subplots(rows=n_copies, cols=1, subplot_titles=titles)

        # --- 3. Signal Preparation & 4. Plotting ---
        for i, recording in enumerate(recordings):
            raw = recording.data
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

            fig.add_trace(go.Scatter(x=times, y=ch_data[0, :] if ch_data.ndim > 1 else ch_data, mode="lines", name=titles[i]), row=i + 1, col=1)

            # Update subplot titles to include subject and channel
            fig.layout.annotations[i].text = f"{titles[i]} - Subject {recording.subject_id}, Channel {ch_name}"

        fig.update_layout(height=self.DEFAULT_ROW_HEIGHT * n_copies, title_text="Raw Augmentation Variety Check", template="plotly_white")
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude")

        self._handle_output(fig, "raw_augmentation_interactive.html", run_ctx)

    def visualize_epochs(self, data: EpochPreprocessedDTO, run_ctx: RunContext) -> None:
        """
        Visualizes Event-Related Potential (ERP) averages using interactive line plots.

        Follows these sequential steps:
        1. ERP Calculation: Averages data across all epochs for the first recording.
        2. Channel Selection: Identifies a subset of channels for clear visualization.
        3. Trace Generation: Creates interactive Scatter traces for each selected channel.
        4. Final Rendering: Applies layout styling and saves the HTML report.

        Args:
            data (EpochPreprocessedDTO): Preprocessed epoched data.
            run_ctx (RunContext): Pipeline execution context.
        """
        if not self._config.visualize_epochs or not data.data:
            return

        log = run_ctx.logger.for_step("PLOTLY_VISUALISATION_EPOCHS")
        log.info("Visualizing ERP (interactive) of epoched data...", VerbosityLevel.QUIET)

        # --- 1. ERP Calculation ---
        recording = data.data[0]
        epochs = recording.data

        if hasattr(epochs, "get_data"):
            data_arr = epochs.get_data(copy=False)
            erp = np.mean(data_arr, axis=0)  # (channels, times)
            times = epochs.times

            # --- 2. Channel Selection & 3. Trace Generation ---
            fig = go.Figure()
            # Plot first few channels to avoid cluttering
            for i in range(min(5, erp.shape[0])):
                fig.add_trace(go.Scatter(x=times, y=erp[i], mode="lines", name=f"Channel {i}"))

            # --- 4. Final Rendering ---
            fig.update_layout(title=f"ERP Average - Subject {recording.subject_id}", xaxis_title="Time (s)", yaxis_title="Amplitude (uV)", template="plotly_white")
            self._handle_output(fig, "epoch_erp_interactive.html", run_ctx)

    def visualize_augmentation(self, data: DatasetSplitDTO, run_ctx: RunContext, copies_per_sample: int = 0) -> None:
        """
        Visualizes augmented data comparison for epoched samples using Plotly subplots.

        This method facilitates variety checks for augmentation strategies:
        1. Index Mapping: Identifies original samples and their corresponding augmented copies.
        2. Subplot Preparation: Sets up the figure layout based on the number of copies.
        3. Data Plotting: Renders the time-series for comparison.

        Args:
            data (DatasetSplitDTO): Dataset containing generated folds and augmented samples.
            run_ctx (RunContext): Pipeline execution context.
            copies_per_sample (int): Number of augmented copies per sample.
        """
        if not self._config.visualize_augmentation or not data.folds:
            return

        log = run_ctx.logger.for_step("PLOTLY_VISUALISATION_AUGMENTATION")
        log.info("Visualizing augmented samples (interactive)...", VerbosityLevel.QUIET)

        # --- 1. Index Mapping ---
        fold = data.folds[0]
        recording = fold.train_data.data[0]
        x = recording.data

        if isinstance(x, np.ndarray) and x.ndim == 3:
            n_original_samples = x.shape[0] // (1 + copies_per_sample)
            # We want to show the first original sample and its augmented copies
            indices_to_plot = [0] + [(i + 1) * n_original_samples for i in range(copies_per_sample)]

            # --- 2. Subplot Preparation ---
            fig = make_subplots(rows=len(indices_to_plot), cols=1, subplot_titles=["Original Sample"] + [f"Augmented Copy {i}" for i in range(1, len(indices_to_plot))])

            # --- 3. Data Plotting ---
            for i, idx in enumerate(indices_to_plot):
                fig.add_trace(go.Scatter(y=x[idx, 0, :], mode="lines", name="Original" if i == 0 else f"Copy {i}"), row=i + 1, col=1)

            fig.update_layout(height=self.AUGMENTATION_ROW_HEIGHT * len(indices_to_plot), title_text="Augmentation Variety Check", showlegend=False)
            self._handle_output(fig, "augmentation_interactive.html", run_ctx)

    def visualize_evaluation(self, data: EvaluationResultDTO, run_ctx: RunContext, model_name: str) -> None:
        """
        Visualizes evaluation results with interactive heatmaps and metrics charts.

        Generates two distinct visualizations:
        1. Confusion Matrix: An interactive heatmap showing classification performance.
        2. Metrics Overview: A bar chart comparing aggregate performance metrics.

        Args:
            data (EvaluationResultDTO): Contains classification metrics and confusion matrix.
            run_ctx (RunContext): Pipeline execution context.
            model_name (str): Name of the model being evaluated.
        """
        if not self._config.visualize_evaluation:
            return

        if not data.confusion_matrix:
            return

        log = run_ctx.logger.for_step("PLOTLY_VISUALISATION_EVALUATION")
        log.info(f"Generating interactive evaluation report for {model_name}...", VerbosityLevel.QUIET)

        # --- 1. Confusion Matrix ---
        z = data.confusion_matrix
        fig_cm = px.imshow(z, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"), title=f"Confusion Matrix: {model_name}")

        # --- 2. Metrics Bar Chart ---
        metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in data.metrics.items()])
        fig_metrics = px.bar(metrics_df, x="Metric", y="Value", color="Metric", title="Aggregate Metrics", range_y=[0, 1.1])

        self._handle_output(fig_cm, f"evaluation_{model_name.lower()}_cm_interactive.html", run_ctx)
        self._handle_output(fig_metrics, f"evaluation_{model_name.lower()}_metrics_interactive.html", run_ctx)

    def _handle_output(self, fig: object, filename: str, run_ctx: RunContext) -> None:
        """Saves the plotly figure as an interactive HTML file."""
        if not self._config.save_plots:
            return

        log = run_ctx.logger.for_step("SAVE_PLOTS")
        output_dir = run_ctx.output_dir
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / filename

        # fig.write_html is the plotly method
        if hasattr(fig, "write_html"):
            fig.write_html(str(save_path))
            log.info(f"Interactive plot saved to: {save_path}", VerbosityLevel.DETAILED)
        else:
            log.warning(f"Object passed to _handle_output is not a Plotly figure: {type(fig)}")
