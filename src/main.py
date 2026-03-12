import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from pipeline.experiment_pipeline import ExperimentPipeline
from src.pipeline.run_context_factory import RunContextFactory
from src.pipeline.stages.config_validation_stage import ConfigValidationStage
from src.pipeline.stages.data_loading_stage import DataLoadingStage
from src.pipeline.stages.raw_data_validation_stage import RawDataValidationStage
from src.pipeline.stages.preprocessing_stage import PreprocessingStage
from src.pipeline.stages.sample_preparation_stage import SamplePreparationStage
from src.pipeline.stages.augmentation_stage import AugmentationStage
from src.pipeline.stages.model_training_stage import ModelTrainingStage
from src.pipeline.stages.evaluation_stage import EvaluationStage
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.evaluation.evaluation_input_dto import EvaluationInputDTO
from src.types.dto.load.data_loading_input_dto import DataLoadingInputDTO
from src.types.dto.preprocessing.preprocessing_input_dto import PreprocessingInputDTO
from src.types.dto.training.sample_preparation_input_dto import SamplePreparationInputDTO
from src.types.dto.training.training_input_dto import TrainingInputDTO

from src.impl.data_loader.dummy_data_loader import DummyLoader

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="../configs/config")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))

    log.info("Experiment start")

    dl = DummyLoader()
    # ex = ExperimentPipeline(
    #     ConfigValidationStage([]),
    #     RunContextFactory(),
    #     DataLoadingStage(DummyLoader())
    # )



if __name__ == "__main__":
    my_app()