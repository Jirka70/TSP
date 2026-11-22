"""Model input contains only EEG / CSD channels, never stim or EOG channels."""

from datetime import datetime, timezone
from unittest.mock import Mock

import mne
import numpy as np
import pytest

from src.impl.model.machine_learning.final_sklearn_trainer import FinalSklearnTrainer
from src.impl.model.util.extract.brain_channel_signal import get_brain_channel_signal
from src.pipeline.context.run_context import RunContext
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.config.model.model_config import SklearnModelConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO

EEG_CHANNEL_COUNT = 4
SAMPLE_COUNT = 64


def create_epochs(channel_types: list[str], epoch_count: int = 6) -> mne.EpochsArray:
    random_generator = np.random.default_rng(0)
    channel_names = [f"channel_{channel_index}" for channel_index in range(len(channel_types))]
    info = mne.create_info(channel_names, sfreq=128.0, ch_types=channel_types)
    signal = random_generator.standard_normal((epoch_count, len(channel_types), SAMPLE_COUNT)) * 1e-5
    labels = np.array([1, 2] * (epoch_count // 2))
    events = np.column_stack([np.arange(epoch_count) * SAMPLE_COUNT, np.zeros(epoch_count, dtype=int), labels])
    return mne.EpochsArray(signal, info, events=events, event_id={"left_hand": 1, "right_hand": 2}, verbose=False)


def create_run_context() -> RunContext:
    return RunContext(
        run_id="brain-channel-test",
        started_at=datetime.now(timezone.utc),
        raw_preprocessing_backend="default",
        paradigm_backend="default",
        epoch_preprocessing_backend="default",
        augmentation_backend=None,
        experiment_name="brain-channel-test",
        pipeline_name="training",
        git_commit_hash=None,
        logger=Mock(spec=PipelineLogger),
    )


def test_keeps_only_eeg_channels_in_original_order() -> None:
    epochs = create_epochs(["eeg", "stim", "eeg", "eog"])

    signal = get_brain_channel_signal(epochs)

    assert signal.shape == (6, 2, SAMPLE_COUNT)
    assert np.array_equal(signal, epochs.get_data(picks=[0, 2]))


def test_keeps_csd_channels_after_csd_re_referencing() -> None:
    epochs = create_epochs(["csd", "csd", "stim"])

    signal = get_brain_channel_signal(epochs)

    assert signal.shape == (6, 2, SAMPLE_COUNT)


def test_keeps_channels_marked_as_bad() -> None:
    epochs = create_epochs(["eeg", "eeg", "stim"])
    epochs.info["bads"] = ["channel_1"]

    signal = get_brain_channel_signal(epochs)

    assert signal.shape == (6, 2, SAMPLE_COUNT)


def test_rejects_epochs_without_brain_channels() -> None:
    epochs = create_epochs(["stim", "eog"])

    with pytest.raises(ValueError, match="no EEG or CSD channel"):
        get_brain_channel_signal(epochs)


def test_final_sklearn_model_is_trained_on_eeg_channels_only() -> None:
    epochs = create_epochs(["eeg"] * EEG_CHANNEL_COUNT + ["eog", "stim"], epoch_count=12)
    recording = RecordingDTO(data=epochs, dataset_name="test", subject_id=1, session_id="0", run_id="0")
    model_configuration = SklearnModelConfig(backend="sklearn", model_name="riemannian_lda", fold_training=False)
    training_input = FinalTrainingInputDTO(
        config=model_configuration,
        training_data=EpochPreprocessedDTO(data=[recording]),
    )

    trained_model = FinalSklearnTrainer().run(training_input, create_run_context()).data.trained_model
    predictions = trained_model.model.predict(get_brain_channel_signal(epochs))

    assert predictions.shape == (12,)
