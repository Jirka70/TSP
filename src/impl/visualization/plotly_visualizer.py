import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pipeline.context.run_context import RunContext
from src.types.dto.config.visualization_config import VisualizationConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.evaluation.evaluation_result_dto import EvaluationResultDTO
from src.types.dto.raw_preprocessing.raw_preprocessed_dto import RawPreprocessedDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO
from src.types.interfaces.visualizer import IVisualizer

log = logging.getLogger(__name__)


class PlotlyVisualizer(IVisualizer):
    """Implementation of IVisualizer using Plotly for interactive HTML reports."""

    def __init__(self, config: VisualizationConfig) -> None:
        """Initializes the visualizer with configuration.

        Args:
            config (VisualizationConfig): Configuration for visualization.
        """
        self._config = config

    def visualize_raw(self, data: RawPreprocessedDTO, run_ctx: RunContext) -> None:
        """Visualizes PSD of the first recording using Plotly."""
        if not self._config.visualize_raw or not data.data:
            return

        log.info("Visualizing PSD (interactive) of preprocessed raw data...")
        recording = data.data[0]
        raw = recording.data

        # Determine safe n_fft
        n_times = int(raw.n_times) if hasattr(raw, "n_times") else 0
        target_n_fft = self._config.n_fft

        if n_times > 0:
            # Ensure n_fft is a power of 2 and <= n_times
            n_fft = 1 << (min(n_times, target_n_fft).bit_length() - 1)
        else:
            n_fft = target_n_fft

        if hasattr(raw, "compute_psd"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*nperseg.*")
                psd = raw.compute_psd(fmax=50, n_fft=n_fft, n_per_seg=n_fft, verbose=False)
                data_arr, freqs = psd.get_data(return_freqs=True)

            # Take average across channels for simplicity in the main plot
            psd_mean = np.mean(data_arr, axis=0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs, y=psd_mean, mode="lines", name="Mean PSD"))
            fig.update_layout(title=f"Power Spectral Density - Subject {recording.subject_id}", xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", template="plotly_white")
            self._handle_output(fig, "raw_psd_interactive.html", run_ctx)

    def visualize_epochs(self, data: EpochPreprocessedDTO, run_ctx: RunContext) -> None:
        """Visualizes ERP (average) of the epoched data using Plotly."""
        if not self._config.visualize_epochs or not data.data:
            return

        log.info("Visualizing ERP (interactive) of epoched data...")
        recording = data.data[0]
        epochs = recording.data

        if hasattr(epochs, "get_data"):
            data_arr = epochs.get_data(copy=False)
            erp = np.mean(data_arr, axis=0)  # (channels, times)
            times = epochs.times

            fig = go.Figure()
            # Plot first few channels
            for i in range(min(5, erp.shape[0])):
                fig.add_trace(go.Scatter(x=times, y=erp[i], mode="lines", name=f"Channel {i}"))

            fig.update_layout(title=f"ERP Average - Subject {recording.subject_id}", xaxis_title="Time (s)", yaxis_title="Amplitude (uV)", template="plotly_white")
            self._handle_output(fig, "epoch_erp_interactive.html", run_ctx)

    def visualize_augmentation(self, data: DatasetSplitDTO, run_ctx: RunContext, copies_per_sample: int = 0) -> None:
        """Visualizes augmented data comparison using Plotly."""
        if not self._config.visualize_augmentation or not data.folds:
            return

        log.info("Visualizing augmented samples (interactive)...")
        fold = data.folds[0]
        recording = fold.train_data.data[0]
        x = recording.data

        if isinstance(x, np.ndarray) and x.ndim == 3:
            n_original_samples = x.shape[0] // (1 + copies_per_sample)

            # We want to show the first original sample and its augmented copies
            indices_to_plot = [0] + [(i + 1) * n_original_samples for i in range(copies_per_sample)]

            fig = make_subplots(rows=len(indices_to_plot), cols=1, subplot_titles=["Original Sample"] + [f"Augmented Copy {i}" for i in range(1, len(indices_to_plot))])

            for i, idx in enumerate(indices_to_plot):
                fig.add_trace(go.Scatter(y=x[idx, 0, :], mode="lines", name="Original" if i == 0 else f"Copy {i}"), row=i + 1, col=1)

            fig.update_layout(height=200 * len(indices_to_plot), title_text="Augmentation Variety Check", showlegend=False)
            self._handle_output(fig, "augmentation_interactive.html", run_ctx)

    def visualize_evaluation(self, data: EvaluationResultDTO, run_ctx: RunContext, model_name: str) -> None:
        """Visualizes evaluation results with interactive heatmaps and charts."""
        if not self._config.visualize_evaluation:
            return

        if not data.confusion_matrix:
            return

        log.info(f"Generating interactive evaluation report for {model_name}...")

        # 1. Confusion Matrix
        z = data.confusion_matrix
        fig_cm = px.imshow(z, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"), title=f"Confusion Matrix: {model_name}")

        # 2. Metrics Bar Chart
        metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in data.metrics.items()])
        fig_metrics = px.bar(metrics_df, x="Metric", y="Value", color="Metric", title="Aggregate Metrics", range_y=[0, 1.1])

        self._handle_output(fig_cm, f"evaluation_{model_name.lower()}_cm_interactive.html", run_ctx)
        self._handle_output(fig_metrics, f"evaluation_{model_name.lower()}_metrics_interactive.html", run_ctx)

    def _handle_output(self, fig: object, filename: str, run_ctx: RunContext) -> None:
        """Saves the plotly figure as an interactive HTML file."""
        if not self._config.save_plots:
            return

        output_dir = run_ctx.output_dir
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / filename

        # fig.write_html is the plotly method
        if hasattr(fig, "write_html"):
            fig.write_html(str(save_path))
            log.info(f"Interactive plot saved to: {save_path}")
        else:
            log.warning(f"Object passed to _handle_output is not a Plotly figure: {type(fig)}")
