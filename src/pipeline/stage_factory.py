from abc import ABC
from enum import Enum
from typing import Any

from src.types.dto.config.experiment_config import ExperimentConfig

from impl.artifacts_saver.artifacts_saver import ArtifactSaver
from impl.augmentation.dummy_augmentor import DummyAugmentor
from impl.evaluator.dummy_evaluator import DummyEvaluator
from impl.model.dummy_model_trainer import DummyModelTrainer
from impl.preprocessing.dummy_preprocessing import DummyPreprocessing
from impl.split.dummy_splitter import DummySplitter
from impl.epoching.dummy_epoching import DummyEpoching
from impl.data_loader.dummy_data_loader import DummyLoader

from src.types.interfaces.epoching import IEpoching
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.preprocessing import IPreprocessing
from src.types.interfaces.splitter import ISplitter
from src.types.interfaces.augmentor import IAugmentor
from src.types.interfaces.model.model_trainer import IModelTrainer
from src.types.interfaces.evaluator import IEvaluator
from src.types.interfaces.artifact_saver import IArtifactSaver


class StageType(Enum):
    DATA_LOADER = 'data_loader'
    PREPROCESSING = 'preprocessing'
    EPOCHING = 'epoching'
    SPLIT = 'split'
    AUGMENTATION = 'augmentation'
    MODEL_TRAINER = 'model_trainer'
    EVALUATOR = 'evaluator'
    SAVER = 'saver'


class StageFactory:
    _targets: dict[StageType, dict[str | None, type]] = {
        StageType.DATA_LOADER: {
            'eegbci': DummyLoader
        },
        StageType.PREPROCESSING: {
            'mne': DummyPreprocessing
        },
        StageType.EPOCHING: {
            'mne': DummyEpoching,
        },
        StageType.SPLIT: {
            'default': DummySplitter
        },
        StageType.AUGMENTATION: {
            'basic': DummyAugmentor,
            None: DummyAugmentor # :)
        },
        StageType.MODEL_TRAINER: {
            'eegnet': DummyModelTrainer,
        },
        StageType.EVALUATOR: {
            'default': DummyEvaluator
        },
        StageType.SAVER: {
            'default': ArtifactSaver
        }
    }

    _config: ExperimentConfig = None

    def __init__(self, config: ExperimentConfig):
        self._config = config

    def create_data_loader(self) -> IDataLoader:
        return StageFactory._targets[StageType.DATA_LOADER][self._config.dataset.name]()

    def create_preprocessing_stage(self) -> IPreprocessing:
        return StageFactory._targets[StageType.PREPROCESSING][self._config.preprocessing.backend]()

    def create_epoching_stage(self) -> IEpoching:
        return StageFactory._targets[StageType.EPOCHING][self._config.epoching.backend]()

    def create_split_stage(self) -> ISplitter:
        return StageFactory._targets[StageType.SPLIT][self._config.split.backend]()

    def create_augmentation_stage(self) -> IAugmentor:
        return StageFactory._targets[StageType.AUGMENTATION][self._config.augmentation.backend]()

    def create_model_trainer_stage(self) -> IModelTrainer:
        return StageFactory._targets[StageType.MODEL_TRAINER][self._config.model.backend]()

    def create_evaluator_stage(self) -> IEvaluator:
        return StageFactory._targets[StageType.EVALUATOR][self._config.evaluation.backend]()

    def create_saver(self) -> IArtifactSaver:
        return StageFactory._targets[StageType.SAVER][self._config.save_artifacts.backend]()