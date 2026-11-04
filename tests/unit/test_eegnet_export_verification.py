from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy
import pytest
import torch
from pydantic import ValidationError

from src.impl.artifacts_saver.artifacts_saver import ArtifactSaver
from src.impl.model.deep_learning.eegnet.builder.eegnet_builder import build_eegnet_model
from src.impl.model.deep_learning.eegnet.load.eegnet_model_loader import EEGNetModelLoader
from src.impl.save_artifacts.deep_learning.eegnet.eegnet_model_serializer import EEGNetModelSerializer
from src.impl.save_artifacts.machine_learning.sklearn_model_serializer import SklearnModelSerializer
from src.pipeline.context.run_context import RunContext
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.config.model.training_config import TrainingConfig
from src.types.dto.model.learning_dataset import LearningDataset
from src.types.dto.model.trained_model_dto import TrainedModelDTO


@pytest.fixture
def trained_model():
    configuration = EEGNetConfig(
        model_name="eegnet",
        backend="eegnet",
        input_normalization="per_epoch_channel",
        fold_training=False,
        n_classes=2,
        dropout=0.25,
        kernel_length=16,
        temporal_filters=2,
        depth_multiplier=1,
        pointwise_filters=2,
        training=TrainingConfig(epochs=1, batch_size=2, learning_rate=0.001, optimizer="adam", random_state=67),
    )
    model = build_eegnet_model(configuration, (3, 2, 128))
    model.initialize_training(numpy.array([3, 7]))
    model.best_epoch = 0
    model.best_validation_accuracy = 0.5
    signal = numpy.random.default_rng(67).standard_normal((3, 2, 128)).astype(numpy.float32)
    return TrainedModelDTO(
        model=model,
        model_name="eegnet",
        best_epoch=0,
        verification_data=LearningDataset(signal, numpy.array([3, 7, 3])),
    )


@pytest.fixture
def run_context():
    return RunContext(
        run_id="export-test",
        started_at=datetime.now(timezone.utc),
        raw_preprocessing_backend="default",
        paradigm_backend="default",
        epoch_preprocessing_backend="default",
        augmentation_backend=None,
        experiment_name="export-test",
        pipeline_name="training",
        git_commit_hash=None,
        logger=Mock(spec=PipelineLogger),
    )


def test_saves_loads_and_verifies_real_eegnet_without_saving_validation_data(trained_model, run_context, tmp_path):
    serializer = EEGNetModelSerializer()
    saved_artifacts = serializer.save(trained_model, tmp_path / "models", run_context)
    assert saved_artifacts.artifacts[0].path == tmp_path / "models" / "eegnet.pt"
    checkpoint = torch.load(saved_artifacts.artifacts[0].path, map_location="cpu", weights_only=True)
    assert "verification_data" not in checkpoint
    assert "verification_data" not in checkpoint["model_state"]
    assert checkpoint["model_state"]["classes"] == [3, 7]
    restored = EEGNetModelLoader().load(tmp_path, run_context)
    numpy.testing.assert_array_equal(
        restored.predict(trained_model.verification_data.x),
        trained_model.model.predict(trained_model.verification_data.x),
    )


@pytest.mark.parametrize("missing", [None, LearningDataset(numpy.empty((0, 2, 128)), numpy.array([], dtype=int))])
def test_requires_nonempty_validation_data_before_writing(trained_model, run_context, tmp_path, missing):
    invalid_model = replace(trained_model, verification_data=missing)
    with pytest.raises(ValueError, match="verification|validation data"):
        EEGNetModelSerializer().save(invalid_model, tmp_path / "models", run_context)
    assert not (tmp_path / "models").exists()


def test_requires_context_for_existing_loader_before_writing(trained_model, tmp_path):
    with pytest.raises(ValueError, match="RunContext"):
        EEGNetModelSerializer().save(trained_model, tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_schema_rejects_wrong_model_type_before_writing(trained_model, run_context, tmp_path):
    invalid_model = replace(trained_model, model=object())
    with pytest.raises(ValidationError, match="EEGNetModel"):
        EEGNetModelSerializer().save(invalid_model, tmp_path / "models", run_context)
    assert not (tmp_path / "models").exists()


def test_rejects_changed_predictions_after_loading(trained_model, run_context, tmp_path):
    restored = Mock()
    restored.predict.return_value = numpy.array([999, 999, 999])
    restored.predict_class_probability.return_value = trained_model.model.predict_class_probability(
        trained_model.verification_data.x,
    )
    with patch.object(EEGNetModelLoader, "load", return_value=restored) as load:
        with pytest.raises(ValueError, match="Predicted labels changed"):
            EEGNetModelSerializer().save(trained_model, tmp_path / "models", run_context)
    load.assert_called_once_with(tmp_path, run_context)


def test_rejects_changed_probabilities_even_when_labels_match(trained_model, run_context, tmp_path):
    restored = Mock()
    restored.predict.return_value = trained_model.model.predict(trained_model.verification_data.x)
    restored.predict_class_probability.return_value = numpy.full((3, 2), numpy.nan)
    with patch.object(EEGNetModelLoader, "load", return_value=restored):
        with pytest.raises(ValueError, match="Probabilities changed"):
            EEGNetModelSerializer().save(trained_model, tmp_path / "models", run_context)


def test_rejects_broadcastable_probability_shape_mismatch(trained_model, run_context, tmp_path):
    restored = Mock()
    restored.predict.return_value = trained_model.model.predict(trained_model.verification_data.x)
    restored.predict_class_probability.return_value = numpy.zeros((1, 2))
    with patch.object(EEGNetModelLoader, "load", return_value=restored):
        with pytest.raises(ValueError, match="Probabilities changed"):
            EEGNetModelSerializer().save(trained_model, tmp_path / "models", run_context)


def test_forwards_existing_run_context_from_artifact_saver(trained_model, run_context, tmp_path):
    serializer = Mock()
    serializer.supports.return_value = True
    input_data = SimpleNamespace(model_serializer=serializer, trained_model=trained_model)
    ArtifactSaver()._save_model(input_data, tmp_path, run_context)
    serializer.save.assert_called_once_with(
        trained_model=trained_model,
        output_path=tmp_path / "models",
        run_ctx=run_context,
    )


def test_sklearn_serializer_accepts_optional_context_without_validation_data(tmp_path, run_context):
    model = Mock()
    model.get_state_dict.return_value = {"weights": [1, 2]}
    trained = TrainedModelDTO(model=model, model_name="csp_lda")
    saved = SklearnModelSerializer().save(trained, tmp_path / "models", run_context)
    assert saved.artifacts[0].path.exists()
