"""Reject incompatible EEGNet exports before constructing a model."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src.impl.model.deep_learning.eegnet.load.eegnet_model_loader import EEGNetModelLoader
from src.pipeline.context.run_context import RunContext
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.model.deep_learning.eegnet.error.invalid_eegnet_checkpoint_error import InvalidEEGNetCheckpointError


@pytest.mark.parametrize("version", [1, 0, -1, 99, True, False, "2", 2.0, None])
def test_rejects_unsupported_or_invalid_explicit_version(tmp_path: Path, version: object) -> None:
    checkpoint = {"format": "eegnet_checkpoint", "format_version": version}
    _assert_checkpoint_is_rejected_before_model_construction(tmp_path, checkpoint)


def test_rejects_checkpoint_without_version(tmp_path: Path) -> None:
    _assert_checkpoint_is_rejected_before_model_construction(tmp_path, {"format": "eegnet_checkpoint"})


def _assert_checkpoint_is_rejected_before_model_construction(
    artifact_directory: Path,
    checkpoint: dict[str, object],
) -> None:
    model_directory = artifact_directory / "models"
    model_directory.mkdir()
    torch.save(checkpoint, model_directory / "eegnet.pt")
    run_context = Mock(spec=RunContext, logger=Mock(spec=PipelineLogger))

    with patch("src.impl.model.deep_learning.eegnet.load.eegnet_model_loader.create_eegnet_network") as build_network:
        with pytest.raises(InvalidEEGNetCheckpointError, match="version"):
            EEGNetModelLoader().load(artifact_directory, run_context)

    build_network.assert_not_called()
