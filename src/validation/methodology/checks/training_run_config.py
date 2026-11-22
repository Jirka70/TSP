"""Loading of the configuration snapshot saved by the training run of a loaded model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAINING_RUN_CONFIG_FILE_NAME = "config.json"


def find_training_run_config_path(model_path: Path) -> Path:
    """Return where the artifact saver stores the training config of the model at ``model_path``.

    The artifact saver writes the model to ``<run>/models/<model file>`` and the config snapshot
    (``ExperimentConfig.model_dump(mode="json")``) to ``<run>/config.json``.
    """
    return model_path.parent.parent / TRAINING_RUN_CONFIG_FILE_NAME


def load_training_run_config(model_path: Path) -> dict[str, Any] | None:
    """Return the dumped training config of the model at ``model_path``, or ``None`` when it was not saved.

    Raises:
        ValueError: If the config snapshot exists but is not a JSON object.
    """
    config_path = find_training_run_config_path(model_path)
    if not config_path.is_file():
        return None

    try:
        with config_path.open(encoding="utf-8") as config_file:
            training_config = json.load(config_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Training run config '{config_path}' is not valid JSON.") from error

    if not isinstance(training_config, dict):
        raise ValueError(f"Training run config '{config_path}' must contain a JSON object.")
    return training_config
