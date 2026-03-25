import logging

import hydra

from pipeline.stage_factory import StageFactory, StageType
from src.impl.data_loader.MOABBDataLoader import MOABBDataLoader
from src.pipeline.context.run_context import RunContext
from src.pipeline.experiment.experiment_pipeline import ExperimentPipeline
from src.pipeline.pipeline import IPipeline
from src.pipeline.run_context_factory import RunContextFactory
from src.pipeline.training.training_pipeline import TrainingPipeline
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

    sf = StageFactory(ex_conf)

    """dl = sf.create_data_loader()
    preprocessing = sf.create_preprocessing_stage()
    epoching = sf.create_epoching_stage()
    split = sf.create_split_stage()
    augmentation = sf.create_augmentation_stage()
    model_trainer = sf.create_model_trainer_stage()
    evaluator = sf.create_evaluator_stage()
    saver = sf.create_saver()

    run_ctx_factory: RunContextFactory = RunContextFactory()

    pipeline: IPipeline
    run_ctx: RunContext = run_ctx_factory.create(ex_conf, "pepa zetek", "adam mika")
    if ex_conf.mode == Mode.TRAINING.value:
        pipeline = TrainingPipeline(dl, preprocessing, epoching, split, augmentation, model_trainer, evaluator, saver)
    elif ex_conf.mode == Mode.EXPERIMENT.value:
        pipeline = ExperimentPipeline(dl, preprocessing, epoching)
    else:
        raise ValueError(f"Mode {ex_conf.mode} is not supported")

    pipeline.run(ex_conf, run_ctx)"""
    dataloader = MOABBDataLoader()
    data = dataloader.run(ex_conf.dataset, RunContextFactory().create(ex_conf, "", ""))
    print(data)


if __name__ == "__main__":
    my_app()
