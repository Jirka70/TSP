import logging

import hydra

from src.pipeline.experiment_pipeline import ExperimentPipeline
from src.types.dto.config.experiment_config import Mode
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

    dl = ex_conf.dataset.get_instance()
    preprocessing = ex_conf.preprocessing.get_instance()
    epoching = ex_conf.epoching.get_instance()
    split = ex_conf.split.get_instance()
    augmentation = ex_conf.augmentation.get_instance()
    model_trainer = ex_conf.model.get_instance()
    evaluator = ex_conf.evaluation.get_instance()
    saver = ex_conf.save_artifacts.get_instance()



    if ex_conf.mode == Mode.TRAINING.value:
        print("mrdat devky")

    ex = ExperimentPipeline(
        dl,
        preprocessing,
        epoching,
        split,
        augmentation,
        model_trainer,
        evaluator,
        saver
    )

    ex.run(validation_res.config)


if __name__ == "__main__":
    my_app()
