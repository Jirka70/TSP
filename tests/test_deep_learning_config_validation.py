from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.config.model.training_config import TrainingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_eegnet_config() -> dict:
    return {
        "backend": "eegnet",
        "model_name": "eegnet",
        "input_normalization": "per_epoch_channel",
        "n_classes": 2,
        "dropout": 0.25,
        "kernel_length": 16,
        "temporal_filters": 8,
        "depth_multiplier": 2,
        "pointwise_filters": 16,
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.0005,
            "optimizer": "ADAM",
            "random_state": 67,
            "deterministic": True,
        },
    }


def test_valid_eegnet_config_normalizes_optimizer() -> None:
    config = EEGNetConfig.model_validate(_valid_eegnet_config())

    assert config.training.optimizer == "adam"


def test_default_training_yaml_matches_training_config_schema() -> None:
    with (PROJECT_ROOT / "configs/model/training/default.yaml").open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    config = TrainingConfig.model_validate(raw_config)

    assert config.optimizer == "adam"


def test_eegnet_config_requires_matching_pointwise_filters() -> None:
    config = _valid_eegnet_config()
    config["pointwise_filters"] = 15

    with pytest.raises(ValidationError, match="pointwise_filters"):
        EEGNetConfig.model_validate(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 0),
        ("batch_size", 0),
        ("learning_rate", 0),
        ("learning_rate", 1.5),
        ("optimizer", "rmsprop"),
        ("random_state", -1),
    ],
)
def test_training_config_rejects_invalid_deep_learning_hyperparameters(field: str, value: object) -> None:
    config = _valid_eegnet_config()
    config["training"][field] = value

    with pytest.raises(ValidationError):
        EEGNetConfig.model_validate(config)


def test_training_config_rejects_unknown_keys() -> None:
    config = _valid_eegnet_config()
    config["training"]["unknown_key"] = True

    with pytest.raises(ValidationError, match="unknown_key"):
        EEGNetConfig.model_validate(config)


def test_legacy_checkpoint_configuration_does_not_enable_fold_training() -> None:
    configuration = _valid_eegnet_config()
    configuration["fold_training"] = True
    restored_configuration = EEGNetConfig.model_validate(configuration)
    assert "fold_training" not in restored_configuration.model_dump()


@pytest.mark.parametrize("source_name", ["synthetic", "moabb"])
def test_final_eegnet_smoke_configuration_uses_direct_splits(source_name: str) -> None:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        configuration = compose(config_name=f"test/smoke/{source_name}/training_dl_final_only")
    assert configuration.split.backend == "general"
    assert "fold_training" not in configuration.model
    assert EEGNetConfig.model_validate(dict(configuration.model)).backend == "eegnet"
