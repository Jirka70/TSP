"""Repeat May 15 source/preprocessing with current stages, one subject at a time."""

import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import mne
import numpy
import torch
from omegaconf import OmegaConf

from src.pipeline.run_context_factory import RunContextFactory
from src.pipeline.stage_factory import StageFactory
from src.pipeline_logging.setup_logging import setup_pipeline_logging
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.epoch_preprocessing.epoch_preprocessing_input_dto import EpochPreprocessingInputDTO
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.paradigm.paradigm_input_dto import ParadigmInputDTO
from src.types.dto.raw_augmentation.raw_augmentation_input_dto import RawAugmentationInputDTO
from src.types.dto.raw_preprocessing.raw_preprocessing_input_dto import RawPreprocessingInputDTO
from src.types.dto.save_artifacts.save_artifacts_input_dto import SaveArtifactsInputDTO
from src.types.dto.split.split_input_dto import SplitInputDTO
from src.impl.model.util.extract.extract_learning_data import extract_learning_data

DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = DIRECTORY.parents[1] / "outputs/dl-performance-2026-11-04"


def convert_processed_recording_to_arrays(recording):
    labels = recording.data.events[:, -1].copy()
    metadata = dict(recording.metadata)
    metadata["labels"] = labels
    return dataclasses.replace(recording, data=recording.data.get_data(copy=True), metadata=metadata)


def preprocess_one_subject(configuration, factory, run_context, subject_identifier):
    subject_configuration = configuration.source.model_copy(update={"subject_ids": [subject_identifier]})
    loaded = factory.create_data_loader().run(subject_configuration, run_context)
    raw = factory.create_raw_preprocessing_stage().run(
        RawPreprocessingInputDTO(configuration.raw_preprocessing, loaded.data),
        run_context,
    )
    del loaded
    augmented = factory.create_raw_augmentation_stage().run(
        RawAugmentationInputDTO(configuration.raw_augmentation, raw.data),
        run_context,
    )
    del raw
    epochs = factory.create_paradigm_stage().run(ParadigmInputDTO(configuration.paradigm, augmented.data), run_context)
    del augmented
    processed = factory.create_epoch_preprocessing_stage().run(
        EpochPreprocessingInputDTO(configuration.epoch_preprocessing, epochs.data),
        run_context,
    )
    return [convert_processed_recording_to_arrays(recording) for recording in processed.data.data]


def describe_dataset(dataset):
    learning_data = extract_learning_data(dataset)
    labels, counts = numpy.unique(learning_data.y, return_counts=True)
    return {
        "shape": list(learning_data.x.shape),
        "class_counts": dict(zip(labels.tolist(), counts.tolist())),
        "signal_standard_deviation": float(numpy.std(learning_data.x)),
        "signal_sha256": hashlib.sha256(learning_data.x.tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(learning_data.y.tobytes()).hexdigest(),
    }


def run_benchmark():
    started = time.perf_counter()
    configuration = ExperimentConfig.model_validate(
        OmegaConf.to_container(OmegaConf.load(DIRECTORY / "current_run.yaml"))
    )
    setup_pipeline_logging(configuration.logging)
    mne.set_log_level("WARNING")
    factory = StageFactory(configuration)
    run_context = RunContextFactory().create(configuration, "may-15-source-repeat", "streamed-current-training")
    recordings = []
    for subject_identifier in configuration.source.subject_ids:
        print(f"BENCHMARK preprocessing subject {subject_identifier}", flush=True)
        recordings.extend(preprocess_one_subject(configuration, factory, run_context, subject_identifier))
        gc.collect()
    processed = EpochPreprocessedDTO(data=recordings)
    preprocessing_seconds = time.perf_counter() - started
    print(f"BENCHMARK preprocessing complete: {len(recordings)} recordings", flush=True)
    split = factory.create_split_stage().run(SplitInputDTO(configuration.split, processed), run_context).data
    dataset_descriptions = {name: describe_dataset(getattr(split, name)) for name in ("train", "validation", "test")}
    augmented = (
        factory.create_augmentation_stage()
        .run(
            AugmentationInputDTO(configuration.augmentation, split),
            run_context,
        )
        .data
    )
    dataset_descriptions["augmented_train"] = describe_dataset(augmented.train)
    print(f"BENCHMARK datasets: {json.dumps(dataset_descriptions)}", flush=True)
    (OUTPUT_DIRECTORY / "dataset_descriptions.json").write_text(json.dumps(dataset_descriptions, indent=2))
    training_started = time.perf_counter()
    trained = (
        factory.create_final_trainer_stage()
        .run(
            FinalTrainingInputDTO(configuration.model, augmented.train, augmented.validation),
            run_context,
        )
        .data.trained_model
    )
    training_seconds = time.perf_counter() - training_started
    evaluation = (
        factory.create_evaluator_stage()
        .run(
            EvaluationInputDTO(configuration.evaluation, trained, augmented.test),
            run_context,
        )
        .data
    )
    factory.create_saver().run(
        SaveArtifactsInputDTO(
            config=configuration.save_artifacts,
            experiment_config=configuration,
            output_path=OUTPUT_DIRECTORY / "artifacts",
            model_serializer=factory.create_model_serializer(),
            trained_model=trained,
            evaluation_result=evaluation,
        ),
        run_context,
    )
    result = {
        "configuration": configuration.model_dump(mode="json"),
        "datasets": dataset_descriptions,
        "best_epoch_zero_based": trained.best_epoch,
        "best_validation_accuracy": trained.best_validation_metric_value,
        "test_metrics": evaluation.metrics,
        "confusion_matrix": evaluation.confusion_matrix,
        "history": dataclasses.asdict(trained.history),
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": training_seconds,
        "total_seconds": time.perf_counter() - started,
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "mne", "moabb", "braindecode", "numpy")
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "threads": torch.get_num_threads(),
        "git_commit": run_context.git_commit_hash,
        "note": "Current uncommitted working tree; sequential subject preprocessing, same stage implementations.",
    }
    (OUTPUT_DIRECTORY / "results.json").write_text(json.dumps(result, indent=2))
    print(f"BENCHMARK RESULT: {json.dumps(result['test_metrics'])}", flush=True)


if __name__ == "__main__":
    run_benchmark()
