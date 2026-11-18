"""Repeat the saved November 4 experiments with cached, identical input recordings."""

import argparse
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/dl-refactor-2026-11-07"


def load_previous_experiment(case: str) -> ModuleType:
    script_name = "run_benchmark.py"
    if case == "subject-8":
        script_name = "run_subject_8_benchmark.py"
    script_path = ROOT / "reviews/dl-performance-2026-11-04" / script_name
    module_description = importlib.util.spec_from_file_location("previous_experiment", script_path)
    if module_description is None or module_description.loader is None:
        raise RuntimeError("Cannot load the previous experiment script.")
    experiment = importlib.util.module_from_spec(module_description)
    module_description.loader.exec_module(experiment)
    return experiment


def run_comparison(phase: str, case: str) -> None:
    source_directory = OUTPUT / "final_source"
    if phase == "baseline":
        source_directory = OUTPUT / "baseline_source"
    sys.path.insert(0, str(source_directory))

    import numpy
    import torch

    from src.impl.model.deep_learning.eegnet.load.eegnet_model_loader import EEGNetModelLoader
    from src.impl.model.util.extract.extract_learning_data import extract_learning_data
    from src.pipeline.context.run_context import RunContext
    from src.pipeline.stage_factory import StageFactory
    from src.types.dto.config.experiment_config import ExperimentConfig
    from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
    from src.types.dto.load.recording import RecordingDTO

    torch.set_num_threads(2 if case == "subject-8" else 4)
    experiment = load_previous_experiment(case)
    output_directory = OUTPUT / phase / case
    output_directory.mkdir(parents=True, exist_ok=True)
    setattr(experiment, "OUTPUT_DIRECTORY", output_directory)
    original_preprocessing = experiment.preprocess_one_subject
    original_description = experiment.describe_dataset
    described_datasets: list[EpochPreprocessedDTO] = []
    contexts: list[RunContext] = []

    def cached_preprocessing(
        configuration: ExperimentConfig,
        factory: StageFactory,
        run_context: RunContext,
        subject_identifier: int,
    ) -> list[RecordingDTO]:
        cache_directory = OUTPUT / "recordings"
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_path = cache_directory / f"subject-{subject_identifier}.pickle"
        contexts.append(run_context)
        if cache_path.exists():
            with cache_path.open("rb") as cache_file:
                cached_recordings: object = pickle.load(cache_file)
            return validate_recording_list(cached_recordings)
        recordings = validate_recording_list(
            original_preprocessing(configuration, factory, run_context, subject_identifier)
        )
        with cache_path.open("wb") as cache_file:
            pickle.dump(recordings, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        return recordings

    def capture_description(dataset: EpochPreprocessedDTO) -> dict[str, object]:
        described_datasets.append(dataset)
        description: object = original_description(dataset)
        if not isinstance(description, dict):
            raise TypeError("The previous experiment must return a dataset description dictionary.")
        dataset_description: dict[str, object] = {}
        for key, value in description.items():
            if not isinstance(key, str):
                raise TypeError("Dataset description keys must be strings.")
            dataset_description[key] = value
        return dataset_description

    def validate_recording_list(value: object) -> list[RecordingDTO]:
        if not isinstance(value, list):
            raise TypeError("Recording cache must contain a list.")
        recordings: list[RecordingDTO] = []
        for recording in value:
            if not isinstance(recording, RecordingDTO):
                raise TypeError("Recording cache must contain only RecordingDTO values.")
            recordings.append(recording)
        return recordings

    setattr(experiment, "preprocess_one_subject", cached_preprocessing)
    setattr(experiment, "describe_dataset", capture_description)
    experiment.run_benchmark()

    test_dataset = extract_learning_data(described_datasets[2])
    checkpoint_paths = list((output_directory / "artifacts").glob("*/models/eegnet.pt"))
    if len(checkpoint_paths) != 1:
        raise RuntimeError("Expected exactly one saved EEGNet checkpoint.")
    model = EEGNetModelLoader().load(checkpoint_paths[0].parent.parent, contexts[0])
    reference_path = output_directory / "reference_predictions.npz"
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("reference_predictions.py")), case, str(reference_path)],
        check=True,
    )
    numpy.savez(
        output_directory / "test_predictions.npz",
        labels=model.predict(test_dataset.x),
        probabilities=model.predict_class_probability(test_dataset.x),
        expected_labels=test_dataset.y,
    )
    provenance = {
        "phase": phase,
        "case": case,
        "source_directory": str(source_directory),
        "checkpoint": str(checkpoint_paths[0]),
    }
    (output_directory / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("baseline", "refactored"))
    parser.add_argument("case", choices=("general", "subject-8"))
    arguments = parser.parse_args()
    run_comparison(arguments.phase, arguments.case)
