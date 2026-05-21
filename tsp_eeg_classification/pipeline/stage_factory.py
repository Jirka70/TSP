from enum import Enum

from tsp_eeg_classification.impl.model.deep_learning.trainer.eegnet_model_trainer import EEGNetModelTrainer
from tsp_eeg_classification.impl.artifacts_saver.artifacts_saver import ArtifactSaver
from tsp_eeg_classification.impl.augmentation.basic_augmentor import BasicAugmentor
from tsp_eeg_classification.impl.augmentation.dummy_augmentor import DummyAugmentor
from tsp_eeg_classification.impl.augmentation.torcheeg_augmentor import TorchEEGAugmentor
from tsp_eeg_classification.impl.data_loader.FilesystemDatasetLoader import FilesystemDatasetLoader

# from tsp_eeg_classification.impl.augmentation.torcheeg_augmentor import TorchEEGAugmentor
from tsp_eeg_classification.impl.data_loader.MOABBDataLoader import MOABBDataLoader
from tsp_eeg_classification.impl.dataset_export.fif_dataset_exporter import FifDatasetExporter
from tsp_eeg_classification.impl.epoch_preprocessing.epoch_preprocessing import EpochPreprocessor
from tsp_eeg_classification.impl.evaluator.standard_evaluator import StandardEvaluator
from tsp_eeg_classification.impl.model.deep_learning.trainer.final_eegnet_trainer import FinalEEGNetTrainer
from tsp_eeg_classification.impl.model.machine_learning.final_sklearn_trainer import FinalSklearnTrainer
from tsp_eeg_classification.impl.model.machine_learning.generic_sklearn_trainer import GenericSklearnTrainer
from tsp_eeg_classification.impl.model.metrics_aggregator import MetricsAggregator
from tsp_eeg_classification.impl.model.model_loader import ModelLoader
from tsp_eeg_classification.impl.save_artifacts.deep_learning.eegnet_model_serializer import EEGNetModelSerializer
from tsp_eeg_classification.impl.save_artifacts.machine_learning.sklearn_model_serializer import SklearnModelSerializer
from tsp_eeg_classification.impl.paradigm.paradigm_preprocessing import ParadigmPreprocessor
from tsp_eeg_classification.impl.raw_augmentation.dummy_raw_augmentor import DummyRawAugmentor
from tsp_eeg_classification.impl.raw_augmentation.torcheeg_raw_augmentor import TorchEEGRawAugmentor
from tsp_eeg_classification.impl.raw_preprocessing.raw_preprocessing import RawPreprocessor
from tsp_eeg_classification.impl.split.basic_splitter import BasicSplitter
from tsp_eeg_classification.impl.split.moabb_splitter import MoabbSplitter
from tsp_eeg_classification.impl.visualization.matplotlib_visualizer import MatplotlibVisualizer
from tsp_eeg_classification.impl.visualization.plotly_visualizer import PlotlyVisualizer
from tsp_eeg_classification.types.dto.config.experiment_config import ExperimentConfig
from tsp_eeg_classification.types.interfaces.artifact_saver import IArtifactSaver
from tsp_eeg_classification.types.interfaces.augmentor import IAugmentor
from tsp_eeg_classification.types.interfaces.data_loader import IDataLoader
from tsp_eeg_classification.types.interfaces.dataset_exporter import IDatasetExporter
from tsp_eeg_classification.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from tsp_eeg_classification.types.interfaces.evaluator import IEvaluator
from tsp_eeg_classification.types.interfaces.metrics_aggregator import IMetricsAggregator
from tsp_eeg_classification.types.interfaces.model.final_trainer import IFinalTrainer
from tsp_eeg_classification.types.interfaces.model.model_loader import IModelLoader
from tsp_eeg_classification.types.interfaces.model.model_serializer import IModelSerializer
from tsp_eeg_classification.types.interfaces.model.model_trainer import IModelTrainer
from tsp_eeg_classification.types.interfaces.paradigm import IParadigm
from tsp_eeg_classification.types.interfaces.raw_augmentor import IRawAugmentor
from tsp_eeg_classification.types.interfaces.raw_preprocessing import IRawPreprocessing
from tsp_eeg_classification.types.interfaces.splitter import ISplitter
from tsp_eeg_classification.types.interfaces.visualizer import IVisualizer


class StageType(Enum):
    DATA_LOADER = "data_loader"
    RAW_PREPROCESSING = "raw_preprocessing"
    RAW_AUGMENTATION = "raw_augmentation"
    PARADIGM = "paradigm"
    EPOCH_PREPROCESSING = "epoch_preprocessing"
    SPLIT = "split"
    AUGMENTATION = "augmentation"
    MODEL_TRAINER = "model_trainer"
    METRICS_AGGREGATOR = "metrics_aggregator"
    FINAL_TRAINER = "final_trainer"
    EVALUATOR = "evaluator"
    SAVER = "saver"
    MODEL_SERIALIZER = "serializer"
    VISUALIZER = "visualizer"
    MODEL_PATH = "model_path"
    DATASET_EXPORT = "dataset_export"


class StageFactory:
    _targets: dict[StageType, dict[str | None, type]] = {
        StageType.DATA_LOADER: {"external": MOABBDataLoader, "filesystem": FilesystemDatasetLoader},
        StageType.RAW_PREPROCESSING: {"default": RawPreprocessor},
        StageType.RAW_AUGMENTATION: {
            "none": DummyRawAugmentor,
            "raw_torcheeg": TorchEEGRawAugmentor,
        },
        StageType.PARADIGM: {"default": ParadigmPreprocessor},
        StageType.EPOCH_PREPROCESSING: {"default": EpochPreprocessor},
        StageType.SPLIT: {
            "basic": BasicSplitter,
            "moabb_within_session": MoabbSplitter,
            "moabb_within_subject": MoabbSplitter,
            "moabb_cross_subject": MoabbSplitter,
            "moabb_cross_session": MoabbSplitter,
        },
        StageType.AUGMENTATION: {
            "basic": BasicAugmentor,
            "torcheeg": DummyAugmentor,
            None: DummyAugmentor,
        },
        StageType.MODEL_TRAINER: {
            "eegnet": EEGNetModelTrainer,
            "sklearn": GenericSklearnTrainer,
        },
        StageType.METRICS_AGGREGATOR: {"default": MetricsAggregator},
        StageType.FINAL_TRAINER: {
            "sklearn": FinalSklearnTrainer,
            "eegnet": FinalEEGNetTrainer
        },
        StageType.EVALUATOR: {
            "default": StandardEvaluator,
        },
        StageType.SAVER: {"default": ArtifactSaver},
        StageType.MODEL_SERIALIZER: {
            "sklearn": SklearnModelSerializer,
            "eegnet": EEGNetModelSerializer,
        },
        StageType.VISUALIZER: {
            "matplotlib": MatplotlibVisualizer,
            "plotly": PlotlyVisualizer,
        },
        StageType.MODEL_PATH: {
            "default": ModelLoader,
        },
        StageType.DATASET_EXPORT: {
            "fif": FifDatasetExporter,
            "none": None,
        },
    }

    _config: ExperimentConfig = None

    def __init__(self, config: ExperimentConfig):
        self._config = config

    def create_data_loader(self) -> IDataLoader:
        return StageFactory._targets[StageType.DATA_LOADER][self._config.source.backend]()

    def create_raw_preprocessing_stage(self) -> IRawPreprocessing:
        return StageFactory._targets[StageType.RAW_PREPROCESSING][self._config.raw_preprocessing.backend]()

    def create_raw_augmentation_stage(self) -> IRawAugmentor:
        return StageFactory._targets[StageType.RAW_AUGMENTATION][self._config.raw_augmentation.backend]()

    def create_paradigm_stage(self) -> IParadigm:
        return StageFactory._targets[StageType.PARADIGM][self._config.paradigm.backend]()

    def create_epoch_preprocessing_stage(self) -> IEpochPreprocessing:
        return StageFactory._targets[StageType.EPOCH_PREPROCESSING][self._config.epoch_preprocessing.backend]()

    def create_split_stage(self) -> ISplitter:
        return StageFactory._targets[StageType.SPLIT][self._config.split.backend]()

    def create_augmentation_stage(self) -> IAugmentor:
        return StageFactory._targets[StageType.AUGMENTATION][self._config.augmentation.backend]()

    def create_model_trainer_stage(self) -> IModelTrainer:
        return StageFactory._targets[StageType.MODEL_TRAINER][self._config.model.backend]()

    def create_metrics_aggregator_stage(self) -> IMetricsAggregator:
        return StageFactory._targets[StageType.METRICS_AGGREGATOR][self._config.metrics_aggregator.backend]()

    def create_final_trainer_stage(self) -> IFinalTrainer:
        return StageFactory._targets[StageType.FINAL_TRAINER][self._config.final_trainer.backend]()

    def create_evaluator_stage(self) -> IEvaluator:
        return StageFactory._targets[StageType.EVALUATOR][self._config.evaluation.backend]()

    def create_visualizer(self) -> IVisualizer:
        return StageFactory._targets[StageType.VISUALIZER][self._config.visualization.backend](self._config.visualization)

    def create_saver(self) -> IArtifactSaver:
        return StageFactory._targets[StageType.SAVER][self._config.save_artifacts.backend]()

    def create_model_serializer(self) -> IModelSerializer:
        return StageFactory._targets[StageType.MODEL_SERIALIZER][self._config.model.backend]()

    def create_model_loader(self) -> IModelLoader:
        return StageFactory._targets[StageType.MODEL_PATH][self._config.model_path.backend]()

    def create_dataset_exporter_stage(self) -> IDatasetExporter:
        return StageFactory._targets[StageType.DATASET_EXPORT][self._config.dataset_export.backend]()
