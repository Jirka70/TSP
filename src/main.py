import logging
from pathlib import Path

import hydra

from src.impl.augmentation.dummy_augmentor import DummyAugmentor
from src.impl.data_loader.dummy_data_loader import DummyLoader
from src.impl.epoching.dummy_epoching import DummyEpoching
from src.impl.model.dummy_model_trainer import DummyModelTrainer
from src.impl.preprocessing.dummy_preprocessing import DummyPreprocessing
from src.impl.split.dummy_splitter import DummySplitter
from src.pipeline.experiment_pipeline import ExperimentPipeline
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.config.epoching_config import EpochingConfig
from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.model.model_config import ModelConfig
from src.types.dto.config.model.training_config import TrainingConfig
from src.types.dto.config.split_config import SplitConfig
from src.types.dto.config.root_config import RootConfig
from validation.impl.root_config_validator import ExperimentConfigValidator

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def my_app(cfg):
    log.info("Experiment start")

    dl = DummyLoader()
    preprocessing = DummyPreprocessing()
    epoching = DummyEpoching()
    split = DummySplitter()
    augmentation = DummyAugmentor()
    model_trainer = DummyModelTrainer()
    ex = ExperimentPipeline(
        dl,
        preprocessing,
        epoching,
        split,
        augmentation,
        model_trainer
    )

    # experiment_config_test = ExperimentConfig()
    v = ExperimentConfigValidator()
    ex_conf = v.validate(cfg)

    ex.run(ex_conf)


if __name__ == "__main__":
    my_app()
