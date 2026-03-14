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
from validation.config_validator import ExperimentConfigValidator

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def my_app(cfg):
    log.info("Experiment start")

    validator = ExperimentConfigValidator()
    validation_res = validator.validate(cfg)
    if validation_res.is_valid:
        log.info("Config successfully validated")
    else:
        log.error("Validation failed")
        return

    ex_conf = validation_res.config

    dl = DummyLoader()
    preprocessing = ex_conf.preprocessing.stage_instance()
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

    ex.run(validation_res.config)

if __name__ == "__main__":
    my_app()
