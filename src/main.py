import logging
import hydra

from src.pipeline.context.run_context import RunContext
from src.pipeline.experiment.experiment_pipeline import ExperimentPipeline
from src.pipeline.pipeline import IPipeline
from src.pipeline.run_context_factory import RunContextFactory
from src.pipeline.stage_factory import StageFactory
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

    dl = sf.create_data_loader()
    raw_preprocessing = sf.create_raw_preprocessing_stage()
    paradigm = sf.create_paradigm_stage()
    epoch_preprocessing = sf.create_epoch_preprocessing_stage()
    evaluator = sf.create_evaluator_stage()
    visualizer = sf.create_visualizer()
    saver = sf.create_saver()
    model_serializer = sf.create_model_serializer()

    run_ctx_factory: RunContextFactory = RunContextFactory()
    run_ctx: RunContext = run_ctx_factory.create(ex_conf, "pepa zetek", "adam mika")

    pipeline: IPipeline

    if ex_conf.mode == Mode.TRAINING.value:
        split = sf.create_split_stage()
        augmentation = sf.create_augmentation_stage()
        model_trainer = sf.create_model_trainer_stage()
        metrics_aggregator = sf.create_metrics_aggregator_stage()
        final_trainer = sf.create_final_trainer_stage()

        pipeline = TrainingPipeline(
            dl, raw_preprocessing, paradigm, epoch_preprocessing,
            split, augmentation, model_trainer, metrics_aggregator,
            final_trainer, evaluator, visualizer, saver, model_serializer
        )

    elif ex_conf.mode == Mode.EXPERIMENT.value:
        model_loader = sf.create_model_loader()

        pipeline = ExperimentPipeline(
            data_loader=dl,
            raw_preprocessing=raw_preprocessing,
            paradigm=paradigm,
            epoch_preprocessing=epoch_preprocessing,
            model_loader=model_loader,
            evaluator=evaluator,
            visualizer=visualizer,
            artifact_saver=saver,
            model_serializer=model_serializer
        )
    else:
        raise ValueError(f"Mode {ex_conf.mode} is not supported")

    pipeline.run(ex_conf, run_ctx)


if __name__ == "__main__":
    my_app()