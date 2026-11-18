"""Inspect channel types and label information in the historical input representation."""

import importlib.util
import json
from pathlib import Path

import mne
import numpy
from omegaconf import OmegaConf
from moabb.datasets import BNCI2014_001

from src.pipeline.run_context_factory import RunContextFactory
from src.pipeline.stage_factory import StageFactory
from src.pipeline_logging.setup_logging import setup_pipeline_logging
from src.types.dto.config.experiment_config import ExperimentConfig

DIRECTORY = Path(__file__).resolve().parent
module_description = importlib.util.spec_from_file_location("benchmark", DIRECTORY / "run_benchmark.py")
benchmark = importlib.util.module_from_spec(module_description)
module_description.loader.exec_module(benchmark)
configuration = ExperimentConfig.model_validate(OmegaConf.to_container(OmegaConf.load(DIRECTORY / "current_run.yaml")))
setup_pipeline_logging(configuration.logging)
mne.set_log_level("WARNING")
factory = StageFactory(configuration)
run_context = RunContextFactory().create(configuration, "channel-diagnostic", "channel-diagnostic")
recordings = benchmark.preprocess_one_subject(configuration, factory, run_context, 1)
raw_recordings = BNCI2014_001().get_data(subjects=[1])
first_raw = next(iter(next(iter(raw_recordings[1].values())).values()))
channel_names = first_raw.ch_names
channel_types = first_raw.get_channel_types()
stimulus_channel = channel_types.index("stim")
signal = numpy.concatenate([recording.data for recording in recordings])
labels = numpy.concatenate([recording.metadata["labels"] for recording in recordings])
stimulus_maxima = signal[:, stimulus_channel, :].max(axis=1)
classes = numpy.unique(labels)
class_ranges = {
    int(label): [float(stimulus_maxima[labels == label].min()), float(stimulus_maxima[labels == label].max())]
    for label in classes
}
threshold = (class_ranges[int(classes[0])][1] + class_ranges[int(classes[1])][0]) / 2
predictions = numpy.where(stimulus_maxima < threshold, classes[0], classes[1])
result = {
    "channel_names": channel_names,
    "channel_types": channel_types,
    "signal_shape": list(signal.shape),
    "stimulus_channel_index": stimulus_channel,
    "stimulus_channel_name": channel_names[stimulus_channel],
    "stimulus_maximum_ranges_by_class": class_ranges,
    "diagnostic_threshold_accuracy": float((predictions == labels).mean()),
    "note": "Subject 1 diagnostic only, not an independent EEG classifier evaluation.",
}
(benchmark.OUTPUT_DIRECTORY / "channel_diagnostic.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
