"""Validated inputs required to save and verify an EEGNet checkpoint."""

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.pipeline.context.run_context import RunContext
from src.types.dto.model.learning_dataset import LearningDataset


class ValidatedEEGNetSerializationInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: EEGNetModel
    verification_data: LearningDataset
    run_context: RunContext

    @model_validator(mode="after")
    def validate_verification_data_have_samples(self) -> Self:
        if self.verification_data.sample_count == 0:
            raise ValueError("EEGNet export verification data contain no samples.")
        return self
