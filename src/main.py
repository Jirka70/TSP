import logging

import hydra

from src.impl.evaluator.dummy_evaluator import DummyEvaluator
from src.impl.model.dummy_model_trainer import DummyModelTrainer
from src.impl.split.dummy_splitter import DummySplitter
from src.pipeline.experiment_pipeline import ExperimentPipeline
from src.validation.config_validator import ExperimentConfigValidator

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

    dl = ex_conf.dataset.stage()
    preprocessing = ex_conf.preprocessing.stage()
    epoching = ex_conf.epoching.stage()
    split = DummySplitter()
    augmentation = ex_conf.augmentation.stage()
    model_trainer = DummyModelTrainer()
    evaluator = DummyEvaluator()

    ex = ExperimentPipeline(
        dl,
        preprocessing,
        epoching,
        split,
        augmentation,
        model_trainer,
        evaluator
    )

    ex.run(validation_res.config)


if __name__ == "__main__":
    my_app()
