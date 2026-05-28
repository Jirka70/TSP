from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from pydantic import ValidationError

from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.pipeline.context.run_context import RunContext
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.model.deep_learning.eegnet.error.invalid_eegnet_checkpoint_error import InvalidEEGNetCheckpointError
from src.types.dto.model.deep_learning.eegnet.load.eegnet_checkpoint import EEGNetCheckpoint
from src.types.dto.model.deep_learning.eegnet.load.eegnet_model_state import EEGNetModelState
from src.types.interfaces.model.model_loader import IModelLoader

STEP_NAME = "EEGNET_LOADER"


def _parse_config(raw_config: dict[str, Any]) -> EEGNetConfig:
    try:
        return EEGNetConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise InvalidEEGNetCheckpointError("Invalid EEGNet config in checkpoint.") from exc


def _required_dict(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)

    if not isinstance(value, dict):
        raise InvalidEEGNetCheckpointError(f"'{key}' must be a dictionary.")

    return value


def _required_str(source: dict[str, Any], key: str) -> str:
    value = source.get(key)

    if not isinstance(value, str) or not value:
        raise InvalidEEGNetCheckpointError(f"'{key}' must be a non-empty string.")

    return value


def _required_int(source: dict[str, Any], key: str) -> int:
    value = source.get(key)

    if not isinstance(value, int):
        raise InvalidEEGNetCheckpointError(f"'{key}' must be an integer.")

    return value


def _optional_int(source: dict[str, Any], key: str) -> int | None:
    value = source.get(key)

    if value is None:
        return None

    if not isinstance(value, int):
        raise InvalidEEGNetCheckpointError(f"'{key}' must be an integer or null.")

    return value


def _optional_float(source: dict[str, Any], key: str) -> float | None:
    value = source.get(key)

    if value is None:
        return None

    if not isinstance(value, (int, float)):
        raise InvalidEEGNetCheckpointError(f"'{key}' must be a number or null.")

    return float(value)


def _parse_input_shape(raw_input_shape: Any) -> tuple[int, int, int]:
    if raw_input_shape is None:
        raise InvalidEEGNetCheckpointError(
            "Missing 'model_state.input_shape'. "
            "The checkpoint cannot be loaded safely without n_epochs, n_channels and n_times."
        )

    if not isinstance(raw_input_shape, (list, tuple)) or len(raw_input_shape) != 3:
        raise InvalidEEGNetCheckpointError(
            "'model_state.input_shape' must be a list or tuple of three integers."
        )

    try:
        n_epochs = int(raw_input_shape[0])
        n_channels = int(raw_input_shape[1])
        n_times = int(raw_input_shape[2])
    except (TypeError, ValueError) as exc:
        raise InvalidEEGNetCheckpointError(
            "'model_state.input_shape' contains non-integer values."
        ) from exc

    input_shape = (n_epochs, n_channels, n_times)

    if any(value <= 0 for value in input_shape):
        raise InvalidEEGNetCheckpointError(
            f"'model_state.input_shape' values must be positive, got {input_shape}."
        )

    return input_shape


def _build_model(model_state: EEGNetModelState) -> EEGNetModel:
    network = create_eegnet_network(
        config=model_state.config,
        shape=model_state.input_shape,
    )

    return EEGNetModel(
        network=network,
        model_name=model_state.model_name,
        config=model_state.config,
        input_shape=model_state.input_shape
    )


def _restore_runtime_state(
        model: EEGNetModel,
        model_state: EEGNetModelState,
) -> None:
    if not hasattr(model, "restore_runtime_state"):
        raise RuntimeError(
            "EEGNetModel must expose restore_runtime_state(...) so the loader can "
            "restore classes and best-validation metadata without mutating private fields."
        )

    model.restore_runtime_state(
        classes=np.array(model_state.classes) if model_state.classes is not None else None,
        best_epoch=model_state.best_epoch,
        best_validation_accuracy=model_state.best_validation_accuracy,
    )

def _parse_model_state(
        model_state: dict[str, Any],
        fallback_model_name: str,
) -> EEGNetModelState:
    model_name = model_state.get("model_name", fallback_model_name)

    if not isinstance(model_name, str) or not model_name:
        raise InvalidEEGNetCheckpointError("'model_state.model_name' must be a non-empty string.")

    config = _parse_config(_required_dict(model_state, "config"))
    input_shape = _parse_input_shape(model_state.get("input_shape"))
    network_state_dict = _required_dict(model_state, "network_state_dict")

    classes = model_state.get("classes")
    if classes is not None and not isinstance(classes, list):
        raise InvalidEEGNetCheckpointError("'model_state.classes' must be a list or null.")

    return EEGNetModelState(
        model_name=model_name,
        config=config,
        input_shape=input_shape,
        network_state_dict=network_state_dict,
        classes=classes,
        best_epoch=_optional_int(model_state, "best_epoch"),
        best_validation_accuracy=_optional_float(
            model_state,
            "best_validation_accuracy",
        ),
    )


class EEGNetModelLoader(IModelLoader):
    CHECKPOINT_FORMAT: Final[str] = "eegnet_checkpoint"
    SUPPORTED_FORMAT_VERSIONS: Final[set[int]] = {1}
    MANIFEST_FILENAME: Final[str] = "manifest.json"
    CHECKPOINT_RELATIVE_PATH: Final[Path] = Path("models") / "eegnet.pt"


    def __init__(self, map_location: str | torch.device = "cpu") -> None:
        self._map_location = map_location

    def load(self, model_path: Path, run_ctx: RunContext) -> Any:
        artifact_dir = model_path.expanduser().resolve()
        log = run_ctx.logger.for_step(STEP_NAME)

        checkpoint_path = self._resolve_checkpoint_path(artifact_dir)

        raw_checkpoint = self._load_checkpoint(checkpoint_path)
        checkpoint = self._parse_checkpoint(raw_checkpoint)

        log.info(f"Loading EEGNet model {checkpoint.model_name} from {model_path}")

        model = _build_model(checkpoint.model_state)
        model.load_network_state_dict(checkpoint.model_state.network_state_dict)
        _restore_runtime_state(model, checkpoint.model_state)

        return model

    def _load_checkpoint(self, model_path: Path) -> dict[str, Any]:
        try:
            checkpoint = torch.load(
                model_path,
                map_location=self._map_location,
                weights_only=True,
            )
        except Exception as exc:
            raise InvalidEEGNetCheckpointError(
                f"Failed to load EEGNet checkpoint from {model_path}"
            ) from exc

        if not isinstance(checkpoint, dict):
            raise InvalidEEGNetCheckpointError(
                f"EEGNet checkpoint must be a dictionary, got {type(checkpoint).__name__}"
            )

        return checkpoint

    def _parse_checkpoint(self, checkpoint: dict[str, Any]) -> EEGNetCheckpoint:
        checkpoint_format = _required_str(checkpoint, "format")

        if checkpoint_format != self.CHECKPOINT_FORMAT:
            raise InvalidEEGNetCheckpointError(
                f"Unsupported checkpoint format '{checkpoint_format}'. "
                f"Expected '{self.CHECKPOINT_FORMAT}'."
            )

        format_version = _required_int(checkpoint, "format_version")
        if format_version not in self.SUPPORTED_FORMAT_VERSIONS:
            raise InvalidEEGNetCheckpointError(
                f"Unsupported EEGNet checkpoint version {format_version}. "
                f"Supported versions: {sorted(self.SUPPORTED_FORMAT_VERSIONS)}"
            )

        model_name = _required_str(checkpoint, "model_name")
        model_state = _parse_model_state(
            _required_dict(checkpoint, "model_state"),
            fallback_model_name=model_name,
        )

        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            raise InvalidEEGNetCheckpointError("'metadata' must be a dictionary when provided.")

        return EEGNetCheckpoint(
            format=checkpoint_format,
            format_version=format_version,
            model_name=model_name,
            model_state=model_state,
            metadata=metadata,
        )

    def _resolve_checkpoint_path(self, artifact_dir: Path) -> Path:
        if not artifact_dir.exists():
            raise FileNotFoundError(f"EEGNet artifact directory does not exist: {artifact_dir}")

        if not artifact_dir.is_dir():
            raise InvalidEEGNetCheckpointError(
                f"EEGNet loader expects artifact directory, got file: {artifact_dir}"
            )

        checkpoint_path = artifact_dir / self.CHECKPOINT_RELATIVE_PATH

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EEGNet checkpoint does not exist: {checkpoint_path}")

        if not checkpoint_path.is_file():
            raise InvalidEEGNetCheckpointError(
                f"EEGNet checkpoint path is not a file: {checkpoint_path}"
            )

        return checkpoint_path
