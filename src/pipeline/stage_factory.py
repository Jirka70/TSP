from enum import Enum

from src.impl.artifacts_saver.artifacts_saver import ArtifactSaver
from src.impl.augmentation.basic_augmentor import BasicAugmentor
from src.impl.augmentation.dummy_augmentor import DummyAugmentor
from src.impl.augmentation.torcheeg_augmentor import TorchEEGAugmentor
from src.impl.data_loader.MOABBDataLoader import MOABBDataLoader
from src.impl.epoch_preprocessing.epoch_preprocessing import EpochPreprocessor
from src.impl.evaluator.dummy_evaluator import DummyEvaluator
from src.impl.model.dummy_model_trainer import DummyModelTrainer
from src.impl.paradigm.paradigm_preprocessing import ParadigmPreprocessor
from src.impl.raw_preprocessing.raw_preprocessing import RawPreprocessor
from src.impl.split.dummy_splitter import DummySplitter
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.interfaces.artifact_saver import IArtifactSaver
from src.types.interfaces.augmentor import IAugmentor
from src.types.interfaces.data_loader import IDataLoader
from src.types.interfaces.epoch_preprocessing import IEpochPreprocessing
from src.types.interfaces.evaluator import IEvaluator
from src.types.interfaces.model.model_trainer import IModelTrainer
from src.types.interfaces.paradigm import IParadigm
from src.types.interfaces.raw_preprocessing import IRawPreprocessing
from src.types.interfaces.splitter import ISplitter



class StageType(Enum):
    DATA_LOADER = "data_loader"
    RAW_PREPROCESSING = "raw_preprocessing"
    PARADIGM = "paradigm"
    EPOCH_PREPROCESSING = "epoch_preprocessing"
    SPLIT = "split"
    AUGMENTATION = "augmentation"
    MODEL_TRAINER = "model_trainer"
    EVALUATOR = "evaluator"
    SAVER = "saver"


class StageFactory:
    _targets: dict[StageType, dict[str | None, type]] = {
        StageType.DATA_LOADER: {"eegbci": MOABBDataLoader},
        StageType.RAW_PREPROCESSING: {"testing": RawPreprocessor},
        StageType.PARADIGM: {"testing": ParadigmPreprocessor},
        StageType.EPOCH_PREPROCESSING: {"testing": EpochPreprocessor},
        StageType.SPLIT: {"default": DummySplitter},
        StageType.AUGMENTATION: {
            "basic": BasicAugmentor,
            "torcheeg": TorchEEGAugmentor,
            None: DummyAugmentor,
        },
        StageType.MODEL_TRAINER: {
            "eegnet": DummyModelTrainer,
        },
        StageType.EVALUATOR: {"default": DummyEvaluator},
        StageType.SAVER: {"default": ArtifactSaver},
    }

    _config: ExperimentConfig = None

    def __init__(self, config: ExperimentConfig):
        self._config = config

    def create_data_loader(self) -> IDataLoader:
        return StageFactory._targets[StageType.DATA_LOADER][self._config.dataset.backend]()

    def create_raw_preprocessing_stage(self) -> IRawPreprocessing:
        return StageFactory._targets[StageType.RAW_PREPROCESSING][self._config.raw_preprocessing.backend]()

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

    def create_evaluator_stage(self) -> IEvaluator:
        return StageFactory._targets[StageType.EVALUATOR][self._config.evaluation.backend]()

    def create_saver(self) -> IArtifactSaver:
        return StageFactory._targets[StageType.SAVER][self._config.save_artifacts.backend]()
