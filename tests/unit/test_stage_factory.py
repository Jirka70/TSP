from types import SimpleNamespace

import pytest

from src.impl.artifacts_saver.artifacts_saver import ArtifactSaver
from src.impl.augmentation.basic_augmentor import BasicAugmentor
from src.impl.augmentation.dummy_augmentor import DummyAugmentor
from src.impl.augmentation.torcheeg_augmentor import TorchEEGAugmentor
from src.impl.data_loader.FilesystemDatasetLoader import FilesystemDatasetLoader
from src.impl.data_loader.MOABBDataLoader import MOABBDataLoader
from src.impl.dataset_export.fif_dataset_exporter import FifDatasetExporter
from src.impl.epoch_preprocessing.epoch_preprocessing import EpochPreprocessor
from src.impl.evaluator.standard_evaluator import StandardEvaluator
from src.impl.model.deep_learning.eegnet.trainer.eegnet_model_trainer import EEGNetModelTrainer
from src.impl.model.deep_learning.eegnet.trainer.final_eegnet_trainer import FinalEEGNetTrainer
from src.impl.model.machine_learning.final_sklearn_trainer import FinalSklearnTrainer
from src.impl.model.machine_learning.generic_sklearn_trainer import GenericSklearnTrainer
from src.impl.model.metrics_aggregator import MetricsAggregator
from src.impl.model.model_loader import ModelLoader
from src.impl.paradigm.paradigm_preprocessing import ParadigmPreprocessor
from src.impl.raw_augmentation.dummy_raw_augmentor import DummyRawAugmentor
from src.impl.raw_augmentation.torcheeg_raw_augmentor import TorchEEGRawAugmentor
from src.impl.raw_preprocessing.raw_preprocessing import RawPreprocessor
from src.impl.save_artifacts.deep_learning.eegnet.eegnet_model_serializer import EEGNetModelSerializer
from src.impl.save_artifacts.machine_learning.sklearn_model_serializer import SklearnModelSerializer
from src.impl.split.basic_splitter import BasicSplitter
from src.impl.split.moabb_splitter import MoabbSplitter
from src.impl.visualization.matplotlib_visualizer import MatplotlibVisualizer
from src.impl.visualization.plotly_visualizer import PlotlyVisualizer
from src.pipeline.stage_factory import StageFactory, StageType


def _stage_config(backend):
    return SimpleNamespace(backend=backend)


def _visualization_config(backend):
    return SimpleNamespace(
        backend=backend,
        visualize_raw=False,
        visualize_raw_augmentation=False,
        visualize_epochs=False,
        visualize_augmentation=False,
        visualize_evaluation=False,
        width=100,
        height=100,
        n_fft=256,
        save_plots=False,
        show_plots=False,
    )


def _factory_config(
    *,
    source="external",
    raw_augmentation="none",
    split="basic",
    augmentation=None,
    model="sklearn",
    final_trainer="sklearn",
    visualization="matplotlib",
):
    return SimpleNamespace(
        source=_stage_config(source),
        raw_preprocessing=_stage_config("default"),
        raw_augmentation=_stage_config(raw_augmentation),
        paradigm=_stage_config("default"),
        epoch_preprocessing=_stage_config("default"),
        split=_stage_config(split),
        augmentation=_stage_config(augmentation),
        model=_stage_config(model),
        metrics_aggregator=_stage_config("default"),
        final_trainer=_stage_config(final_trainer),
        evaluation=_stage_config("default"),
        save_artifacts=_stage_config("default"),
        visualization=_visualization_config(visualization),
        model_path=_stage_config("default"),
        dataset_export=_stage_config("fif"),
    )


@pytest.mark.parametrize(
    ("stage_type", "backend", "expected_class"),
    [
        (StageType.DATA_LOADER, "external", MOABBDataLoader),
        (StageType.DATA_LOADER, "filesystem", FilesystemDatasetLoader),
        (StageType.RAW_PREPROCESSING, "default", RawPreprocessor),
        (StageType.RAW_AUGMENTATION, "none", DummyRawAugmentor),
        (StageType.RAW_AUGMENTATION, "raw_torcheeg", TorchEEGRawAugmentor),
        (StageType.PARADIGM, "default", ParadigmPreprocessor),
        (StageType.EPOCH_PREPROCESSING, "default", EpochPreprocessor),
        (StageType.SPLIT, "basic", BasicSplitter),
        (StageType.SPLIT, "moabb_within_session", MoabbSplitter),
        (StageType.SPLIT, "moabb_within_subject", MoabbSplitter),
        (StageType.SPLIT, "moabb_cross_subject", MoabbSplitter),
        (StageType.SPLIT, "moabb_cross_session", MoabbSplitter),
        (StageType.AUGMENTATION, "basic", BasicAugmentor),
        (StageType.AUGMENTATION, "torcheeg", TorchEEGAugmentor),
        (StageType.AUGMENTATION, None, DummyAugmentor),
        (StageType.MODEL_TRAINER, "eegnet", EEGNetModelTrainer),
        (StageType.MODEL_TRAINER, "sklearn", GenericSklearnTrainer),
        (StageType.METRICS_AGGREGATOR, "default", MetricsAggregator),
        (StageType.FINAL_TRAINER, "sklearn", FinalSklearnTrainer),
        (StageType.FINAL_TRAINER, "eegnet", FinalEEGNetTrainer),
        (StageType.EVALUATOR, "default", StandardEvaluator),
        (StageType.SAVER, "default", ArtifactSaver),
        (StageType.MODEL_SERIALIZER, "sklearn", SklearnModelSerializer),
        (StageType.MODEL_SERIALIZER, "eegnet", EEGNetModelSerializer),
        (StageType.VISUALIZER, "matplotlib", MatplotlibVisualizer),
        (StageType.VISUALIZER, "plotly", PlotlyVisualizer),
        (StageType.MODEL_PATH, "default", ModelLoader),
        (StageType.DATASET_EXPORT, "fif", FifDatasetExporter),
    ],
)
def test_stage_factory_backend_mappings(stage_type, backend, expected_class):
    assert StageFactory._targets[stage_type][backend] is expected_class


def test_stage_factory_creates_expected_default_training_implementations():
    factory = StageFactory(_factory_config())

    assert isinstance(factory.create_data_loader(), MOABBDataLoader)
    assert isinstance(factory.create_raw_preprocessing_stage(), RawPreprocessor)
    assert isinstance(factory.create_raw_augmentation_stage(), DummyRawAugmentor)
    assert isinstance(factory.create_paradigm_stage(), ParadigmPreprocessor)
    assert isinstance(factory.create_epoch_preprocessing_stage(), EpochPreprocessor)
    assert isinstance(factory.create_split_stage(), BasicSplitter)
    assert isinstance(factory.create_augmentation_stage(), DummyAugmentor)
    assert isinstance(factory.create_model_trainer_stage(), GenericSklearnTrainer)
    assert isinstance(factory.create_metrics_aggregator_stage(), MetricsAggregator)
    assert isinstance(factory.create_final_trainer_stage(), FinalSklearnTrainer)
    assert isinstance(factory.create_evaluator_stage(), StandardEvaluator)
    assert isinstance(factory.create_saver(), ArtifactSaver)
    assert isinstance(factory.create_model_serializer(), SklearnModelSerializer)
    assert isinstance(factory.create_visualizer(), MatplotlibVisualizer)
    assert isinstance(factory.create_model_loader(), ModelLoader)
    assert isinstance(factory.create_dataset_exporter_stage(), FifDatasetExporter)


def test_stage_factory_creates_expected_deep_learning_implementations():
    factory = StageFactory(
        _factory_config(
            raw_augmentation="raw_torcheeg",
            split="moabb_within_session",
            augmentation="torcheeg",
            model="eegnet",
            final_trainer="eegnet",
            visualization="plotly",
        )
    )

    assert isinstance(factory.create_raw_augmentation_stage(), TorchEEGRawAugmentor)
    assert isinstance(factory.create_split_stage(), MoabbSplitter)
    assert isinstance(factory.create_augmentation_stage(), TorchEEGAugmentor)
    assert isinstance(factory.create_model_trainer_stage(), EEGNetModelTrainer)
    assert isinstance(factory.create_final_trainer_stage(), FinalEEGNetTrainer)
    assert isinstance(factory.create_model_serializer(), EEGNetModelSerializer)
    assert isinstance(factory.create_visualizer(), PlotlyVisualizer)
