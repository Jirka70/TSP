from dataclasses import dataclass

from src.types.dto.model.deep_learning.eegnet.load.eegnet_model_state import EEGNetModelState


@dataclass(frozen=True)
class EEGNetCheckpoint:
    format: str
    format_version: int
    model_name: str
    model_state: EEGNetModelState
    metadata: dict[str, object]
