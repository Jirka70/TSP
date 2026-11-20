"""Predict with the original v1 loader in a separate process, without migrating checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.types.dto.load.recording import RecordingDTO

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/dl-refactor-2026-11-07"


def read_recordings(cache_path: Path) -> list[RecordingDTO]:
    from src.types.dto.load.recording import RecordingDTO

    with cache_path.open("rb") as cache_file:
        contents: object = pickle.load(cache_file)
    if not isinstance(contents, list):
        raise TypeError("Recording cache must be a list.")
    recordings: list[RecordingDTO] = []
    for recording in contents:
        if not isinstance(recording, RecordingDTO):
            raise TypeError("Recording cache must contain RecordingDTO instances.")
        recordings.append(recording)
    return recordings


def write_predictions(case: str, output_path: Path, source: str) -> None:
    source_directory = OUTPUT / "final_source"
    if source == "original":
        source_directory = OUTPUT / "baseline_source"
    sys.path.insert(0, str(source_directory))

    import numpy
    import torch
    from omegaconf import OmegaConf

    from src.impl.model.deep_learning.eegnet.load.eegnet_model_loader import EEGNetModelLoader
    from src.impl.model.util.extract.extract_learning_data import extract_learning_data
    from src.pipeline.run_context_factory import RunContextFactory
    from src.pipeline.stage_factory import StageFactory
    from src.types.dto.config.experiment_config import ExperimentConfig
    from src.types.dto.config.source.external_dataset_config import ExternalDatasetConfig
    from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
    from src.types.dto.split.split_input_dto import SplitInputDTO

    torch.set_num_threads(2 if case == "subject-8" else 4)
    configuration = ExperimentConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(ROOT / "reviews/dl-performance-2026-11-04/current_run.yaml"))
    )
    if not isinstance(configuration.source, ExternalDatasetConfig):
        raise TypeError("The comparison requires the original external dataset configuration.")
    run_context = RunContextFactory().create(configuration, "dl-reference", "reference-predictions")
    if case == "subject-8":
        test_recordings = read_recordings(OUTPUT / "recordings/subject-8.pickle")
        test_dataset = EpochPreprocessedDTO(data=test_recordings)
    else:
        recordings: list[RecordingDTO] = []
        for subject_identifier in configuration.source.subject_ids:
            recordings.extend(read_recordings(OUTPUT / f"recordings/subject-{subject_identifier}.pickle"))
        split = (
            StageFactory(configuration)
            .create_split_stage()
            .run(SplitInputDTO(configuration.split, EpochPreprocessedDTO(data=recordings)), run_context)
        )
        if split.data.test is None:
            raise ValueError("The original split must provide test data.")
        test_dataset = split.data.test

    learning_dataset = extract_learning_data(test_dataset)
    checkpoint_directory = ROOT / "outputs/dl-performance-2026-11-04"
    if source == "current":
        checkpoint_directory = OUTPUT / "refactored" / case
    elif case == "subject-8":
        checkpoint_directory = checkpoint_directory / "subject-8-held-out"
    reference_directory = ROOT / "outputs/dl-performance-2026-11-04"
    if case == "subject-8":
        reference_directory = reference_directory / "subject-8-held-out"
    expected_dataset = json.loads((reference_directory / "results.json").read_text())["datasets"]["test"]
    if hashlib.sha256(learning_dataset.x.tobytes()).hexdigest() != expected_dataset["signal_sha256"]:
        raise ValueError("Prediction input signal differs from the November 4 test dataset.")
    if hashlib.sha256(learning_dataset.y.tobytes()).hexdigest() != expected_dataset["labels_sha256"]:
        raise ValueError("Prediction input labels differ from the November 4 test dataset.")
    checkpoint_paths = list((checkpoint_directory / "artifacts").glob("*/models/eegnet.pt"))
    if len(checkpoint_paths) != 1:
        raise ValueError("Expected one checkpoint for prediction comparison.")
    model = EEGNetModelLoader().load(checkpoint_paths[0].parent.parent, run_context)
    numpy.savez(
        output_path,
        labels=model.predict(learning_dataset.x),
        probabilities=model.predict_class_probability(learning_dataset.x),
        expected_labels=learning_dataset.y,
    )
    output_path.with_suffix(".json").write_text(
        json.dumps({"source_directory": str(source_directory), "checkpoint": str(checkpoint_paths[0])}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("general", "subject-8"))
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--source", choices=("original", "current"), default="original")
    arguments = parser.parse_args()
    write_predictions(arguments.case, arguments.output_path, arguments.source)
