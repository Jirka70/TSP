"""Load the supported EEGNet checkpoint format and restore inference state."""

import pickle
from pathlib import Path
from typing import Final, TypeGuard

import torch
from pydantic import ValidationError

from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.pipeline.context.run_context import RunContext
from src.types.dto.config.logging.verbosity_level import VerbosityLevel
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.model.deep_learning.eegnet.checkpoint_format import EEGNET_CHECKPOINT_FORMAT, EEGNET_FORMAT_VERSION
from src.types.dto.model.deep_learning.eegnet.error.invalid_eegnet_checkpoint_error import InvalidEEGNetCheckpointError
from src.types.dto.model.deep_learning.eegnet.load.eegnet_checkpoint import EEGNetCheckpoint
from src.types.dto.model.deep_learning.eegnet.load.eegnet_model_state import EEGNetModelState
from src.types.interfaces.model.model_loader import IModelLoader

STEP_NAME = "EEGNET_LOADER"


class EEGNetModelLoader(IModelLoader):
    CHECKPOINT_FORMAT: Final[str] = EEGNET_CHECKPOINT_FORMAT
    SUPPORTED_FORMAT_VERSIONS: Final[frozenset[int]] = frozenset({EEGNET_FORMAT_VERSION})
    CHECKPOINT_RELATIVE_PATH: Final[Path] = Path("models") / "eegnet.pt"

    def __init__(self, map_location: str | torch.device = "cpu") -> None:
        self._map_location = map_location

    def load(self, model_path: Path, run_ctx: RunContext) -> EEGNetModel:
        artifact_directory = model_path.expanduser().resolve()
        logger = run_ctx.logger.for_step(STEP_NAME)
        logger.info(
            f"Starting EEGNet model loading from artifact directory: {artifact_directory}",
            VerbosityLevel.QUIET,
        )
        checkpoint_path = self._resolve_checkpoint_path(artifact_directory)
        checkpoint = self._parse_checkpoint(self._load_checkpoint(checkpoint_path))
        logger.info(
            f"Parsed EEGNet checkpoint: model={checkpoint.model_name}, "
            f"format={checkpoint.format}, version={checkpoint.format_version}",
            VerbosityLevel.DETAILED,
        )

        model_state = checkpoint.model_state
        network = create_eegnet_network(config=model_state.config, shape=model_state.input_shape)
        model = EEGNetModel(
            network=network,
            model_name=model_state.model_name,
            config=model_state.config,
            input_shape=model_state.input_shape,
        )
        model.load_network_state_dict(model_state.network_state_dict)
        model.restore_runtime_state(
            classes=model_state.classes,
            best_epoch=model_state.best_epoch,
            best_validation_accuracy=model_state.best_validation_accuracy,
        )
        logger.info(f"EEGNet model loaded successfully: {checkpoint.model_name}", VerbosityLevel.QUIET)
        return model

    def _load_checkpoint(self, model_path: Path) -> dict[str, object]:
        try:
            checkpoint: object = torch.load(model_path, map_location=self._map_location, weights_only=True)
        except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as error:
            raise InvalidEEGNetCheckpointError(f"Failed to load EEGNet checkpoint from {model_path}") from error
        return _parse_string_keyed_dictionary(checkpoint, "checkpoint")

    def _parse_checkpoint(self, checkpoint: dict[str, object]) -> EEGNetCheckpoint:
        checkpoint_format = _required_str(checkpoint, "format")
        if checkpoint_format != self.CHECKPOINT_FORMAT:
            raise InvalidEEGNetCheckpointError(
                f"Unsupported checkpoint format '{checkpoint_format}'. Expected '{self.CHECKPOINT_FORMAT}'."
            )

        format_version = checkpoint.get("format_version")
        if format_version is None:
            raise InvalidEEGNetCheckpointError("Missing required EEGNet checkpoint 'format_version'.")
        if type(format_version) is not int:
            raise InvalidEEGNetCheckpointError("'format_version' must be an integer.")
        if format_version not in self.SUPPORTED_FORMAT_VERSIONS:
            raise InvalidEEGNetCheckpointError(
                f"Unsupported EEGNet checkpoint version {format_version}. "
                f"Supported versions: {sorted(self.SUPPORTED_FORMAT_VERSIONS)}"
            )

        model_name = _required_str(checkpoint, "model_name")
        return EEGNetCheckpoint(
            format=checkpoint_format,
            format_version=format_version,
            model_name=model_name,
            model_state=_parse_model_state(_required_dict(checkpoint, "model_state"), model_name),
            metadata=_parse_string_keyed_dictionary(checkpoint.get("metadata", {}), "metadata"),
        )

    def _resolve_checkpoint_path(self, artifact_dir: Path) -> Path:
        if not artifact_dir.exists():
            raise FileNotFoundError(f"EEGNet artifact directory does not exist: {artifact_dir}")
        if not artifact_dir.is_dir():
            raise InvalidEEGNetCheckpointError(f"EEGNet loader expects artifact directory, got file: {artifact_dir}")

        checkpoint_path = artifact_dir / self.CHECKPOINT_RELATIVE_PATH
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"EEGNet checkpoint does not exist: {checkpoint_path}")
        if not checkpoint_path.is_file():
            raise InvalidEEGNetCheckpointError(f"EEGNet checkpoint path is not a file: {checkpoint_path}")
        return checkpoint_path


def _parse_model_state(
    model_state: dict[str, object],
    fallback_model_name: str,
) -> EEGNetModelState:
    model_name = model_state.get("model_name", fallback_model_name)
    if not isinstance(model_name, str):
        raise InvalidEEGNetCheckpointError("'model_state.model_name' must be a non-empty string.")
    if not model_name:
        raise InvalidEEGNetCheckpointError("'model_state.model_name' must be a non-empty string.")

    configuration = _parse_config(_required_dict(model_state, "config"))
    input_shape = _parse_input_shape(model_state.get("input_shape"))
    network_state = _parse_network_state(_required_dict(model_state, "network_state_dict"))
    classes = _parse_classes(model_state.get("classes"))
    return EEGNetModelState(
        model_name=model_name,
        config=configuration,
        input_shape=input_shape,
        network_state_dict=network_state,
        classes=classes,
        best_epoch=_optional_int(model_state, "best_epoch"),
        best_validation_accuracy=_optional_float(model_state, "best_validation_accuracy"),
    )


def _parse_config(raw_config: dict[str, object]) -> EEGNetConfig:
    try:
        return EEGNetConfig.model_validate(raw_config)
    except ValidationError as error:
        raise InvalidEEGNetCheckpointError("Invalid EEGNet config in checkpoint.") from error


def _parse_string_keyed_dictionary(value: object, description: str) -> dict[str, object]:
    if not _is_string_keyed_dictionary(value):
        raise InvalidEEGNetCheckpointError(f"'{description}' must be a dictionary with string keys.")
    return value


def _is_string_keyed_dictionary(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in value)


def _required_dict(source: dict[str, object], key: str) -> dict[str, object]:
    return _parse_string_keyed_dictionary(source.get(key), key)


def _required_str(source: dict[str, object], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str):
        raise InvalidEEGNetCheckpointError(f"'{key}' must be a non-empty string.")
    if not value:
        raise InvalidEEGNetCheckpointError(f"'{key}' must be a non-empty string.")
    return value


def _optional_int(source: dict[str, object], key: str) -> int | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise InvalidEEGNetCheckpointError(f"'{key}' must be an integer or null.")
    return value


def _optional_float(source: dict[str, object], key: str) -> float | None:
    value = source.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise InvalidEEGNetCheckpointError(f"'{key}' must be a number or null.")
    return float(value)


def _parse_input_shape(raw_input_shape: object) -> tuple[int, int, int]:
    if raw_input_shape is None:
        raise InvalidEEGNetCheckpointError(
            "Missing 'model_state.input_shape'. "
            "The checkpoint cannot be loaded safely without n_epochs, n_channels and n_times."
        )
    if not isinstance(raw_input_shape, (list, tuple)):
        raise InvalidEEGNetCheckpointError("'model_state.input_shape' must be a list or tuple of three integers.")
    if len(raw_input_shape) != 3:
        raise InvalidEEGNetCheckpointError("'model_state.input_shape' must be a list or tuple of three integers.")

    epoch_count = _parse_shape_dimension(raw_input_shape[0])
    channel_count = _parse_shape_dimension(raw_input_shape[1])
    sample_count = _parse_shape_dimension(raw_input_shape[2])
    input_shape = (epoch_count, channel_count, sample_count)
    if any(dimension <= 0 for dimension in input_shape):
        raise InvalidEEGNetCheckpointError(f"'model_state.input_shape' values must be positive, got {input_shape}.")
    return input_shape


def _parse_shape_dimension(value: object) -> int:
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        raise InvalidEEGNetCheckpointError("'model_state.input_shape' contains non-integer values.")
    try:
        return int(value)
    except (ValueError, OverflowError) as error:
        raise InvalidEEGNetCheckpointError("'model_state.input_shape' contains non-integer values.") from error


def _parse_network_state(raw_state: dict[str, object]) -> dict[str, torch.Tensor]:
    if not _has_only_tensor_parameters(raw_state):
        raise InvalidEEGNetCheckpointError("Network state must contain only tensors.")
    # Keep the original OrderedDict, including PyTorch's module version metadata.
    return raw_state


def _has_only_tensor_parameters(value: dict[str, object]) -> TypeGuard[dict[str, torch.Tensor]]:
    return all(isinstance(parameter, torch.Tensor) for parameter in value.values())


def _parse_classes(value: object) -> list[object] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise InvalidEEGNetCheckpointError("'model_state.classes' must be a list or null.")
    return list(value)
