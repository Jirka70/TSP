import logging
from pathlib import Path

import hydra

from src.impl.data_loader.dummy_data_loader import DummyLoader
from src.impl.epoching.dummy_epoching import DummyEpoching
from src.impl.preprocessing.dummy_preprocessing import DummyPreprocessing
from src.pipeline.experiment_pipeline import ExperimentPipeline
from src.types.dto.config.augmentation_config import AugmentationConfig
from src.types.dto.config.dataset_config import DatasetConfig
from src.types.dto.config.epoching_config import EpochingConfig
from src.types.dto.config.evaluation_config import EvaluationConfig
from src.types.dto.config.experiment_config import ExperimentConfig
from src.types.dto.config.model_config import ModelConfig
from src.types.dto.config.preprocessing_config import PreprocessingConfig

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def my_app(cfg):
    log.info("Experiment start")

    dl = DummyLoader()
    preprocessing = DummyPreprocessing()
    epoching = DummyEpoching()
    ex = ExperimentPipeline(
        dl,
        preprocessing,
        epoching
    )

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
            sampling_rate_hz=128.0,
            rereference="average",
            channel_selection=[""]
        ),
        epoching=EpochingConfig(
            enabled=True,
            backend="mne",
            event_source="annotations",
            event_mapping={},
            event_labels={},
            tmin=20,
            tmax=40,
            baseline=None,
            preload=False,
            reject_by_annotation=False,
            drop_last_incomplete_epoch=False,
            skip_missing_events=True,
            picks=[]
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
        ),
        mode=cfg.mode
    )

    ex.run(experiment_config_mock)


if __name__ == "__main__":
    my_app()
