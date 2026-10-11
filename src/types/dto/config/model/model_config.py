"""Model configuration DTOs."""

# pylint: disable=missing-class-docstring

from typing import Any, Literal

from pydantic import Field, PositiveInt, field_validator, model_validator

from src.types.dto.config.astageconfig import AStageConfig
from src.types.dto.config.model.sklearn_model_parameters import validate_sklearn_model_parameters
from src.types.dto.config.model.training_config import TrainingConfig


class EEGNetConfig(AStageConfig):
    """Configuration for the EEGNet-based deep learning backend."""

    _target_class = "impl.model.dummy_model_trainer.DummyModelTrainer"

    model_name: str

    backend: Literal["eegnet"]

    input_normalization: Literal["none", "per_epoch_channel"]

    fold_training: bool

    n_classes: PositiveInt = Field(ge=2)

    dropout: float = Field(ge=0, le=1)
    kernel_length: PositiveInt
    temporal_filters: PositiveInt
    depth_multiplier: PositiveInt
    pointwise_filters: PositiveInt

    training: TrainingConfig

    @model_validator(mode="after")
    def validate_eegnet_filters(self) -> "EEGNetConfig":
        expected = self.temporal_filters * self.depth_multiplier
        if self.pointwise_filters != expected:
            raise ValueError(f"pointwise_filters ({self.pointwise_filters}) must equal temporal_filters * depth_multiplier ({expected}) for EEGNet.")
        return self


class SklearnModelConfig(AStageConfig):
    """Configuration for sklearn-based model pipelines."""

    fold_training: bool

    backend: Literal["sklearn"]
    model_name: Literal[
        "csp_lda",
        "riemannian_lda",
        "riemannian_svm",
        "riemannian_mdm",
        "riemannian_lr",
        "riemannian_rf",
    ]

    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, v: Any, info: Any) -> dict[str, Any]:
        model_name = info.data.get("model_name")
        return validate_sklearn_model_parameters(model_name, v)
