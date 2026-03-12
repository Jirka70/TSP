import logging
from pathlib import Path

from omegaconf import OmegaConf
import hydra

from src.impl.data_loader.dummy_data_loader import DummyLoader
from src.pipeline.experiment_pipeline import ExperimentPipeline
from src.types.dto.config.augmentation_config import AugmentationConfig
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.model_config import ModelConfig
from src.types.dto.config.preprocessing_config import PreprocessingConfig

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def my_app(cfg):
   # print(OmegaConf.to_yaml(cfg))

    log.info("Experiment start")

    dl = DummyLoader()
    ex = ExperimentPipeline(
        dl
    )

    print(type(cfg))


experiment_config_mock = ExperimentConfig(
    dataset=DatasetConfig(
        name="EEGBCI",
        path=Path("/datasets/eegbci"),
        subject_ids=[1, 2, 3],
        session_ids=None,
        run_ids=[4, 8, 12],
        task="motor_imagery"
    ),
    preprocessing=PreprocessingConfig(
        backend="mne",
        l_freq=8.0,
        h_freq=30.0,
        notch_freq=50.0,
        target_sfreq=128.0,
        rereference="average",
        channel_selection=[""]
    ),
    augmentation=AugmentationConfig(
        enabled=True,
        backend="torch"
    ),
    model=ModelConfig(
        backend="eegnet",
        n_classes=2,
    ),
    evaluation=EvaluationConfig(
        metrics=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
    )
)


    ex.run(ExperimentConfig(
        dataset=DatasetConfig(name="name", ),
        preprocessing=PreprocessingConfig(),
        augmentation=AugmentationConfig(),
        model=ModelConfig(),
        evaluation=EvaluationConfig()
    ))


if __name__ == "__main__":
    my_app()
