import numpy as np
import pytest

from src.impl.augmentation.basic_augmentor import BasicAugmentor, _apply_linear_time_shift
from src.types.dto.augmentation.augmentation_input_dto import AugmentationInputDTO
from src.types.dto.config.augmentation_config import (
    AugmentationConfigBasic,
    ChannelDropoutConfig,
    GaussianNoiseConfig,
    TimeShiftConfig,
)
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.dto.split.dataset_split_dto import DatasetSplitDTO


def test_apply_linear_time_shift_non_circular():
    # 1 channel, 5 time points: [1, 2, 3, 4, 5]
    signal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

    # Shift right by 2 -> edge padded on left: [1, 1, 1, 2, 3] (NOT [4, 5, 1, 2, 3])
    shifted_right = _apply_linear_time_shift(signal, 2)
    expected_right = np.array([[1.0, 1.0, 1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(shifted_right, expected_right)

    # Shift left by 2 -> edge padded on right: [3, 4, 5, 5, 5] (NOT [3, 4, 5, 1, 2])
    shifted_left = _apply_linear_time_shift(signal, -2)
    expected_left = np.array([[3.0, 4.0, 5.0, 5.0, 5.0]])
    np.testing.assert_array_equal(shifted_left, expected_left)


def test_basic_augmentor_replicates_metadata_and_shuffles():
    n_epochs = 4
    n_channels = 2
    n_times = 10
    data = np.arange(n_epochs * n_channels * n_times, dtype=np.float32).reshape(n_epochs, n_channels, n_times)
    labels = np.array([0, 1, 0, 1])
    trial_ids = [101, 102, 103, 104]

    rec = RecordingDTO(
        data=data,
        dataset_name="test_ds",
        subject_id="s1",
        session_id="sess0",
        run_id="run0",
        metadata={"labels": labels, "trial_ids": trial_ids, "sfreq": 250.0},
    )
    preprocessed = EpochPreprocessedDTO(data=[rec])
    split = DatasetSplitDTO(folds=None, train=preprocessed, test=None, validation=None)

    config = AugmentationConfigBasic(
        backend="basic",
        enabled=True,
        copies_per_sample=2,
        random_seed=42,
        gaussian_noise=GaussianNoiseConfig(std=0.0),
        time_shift=TimeShiftConfig(max_samples=0),
        channel_dropout=ChannelDropoutConfig(prob=0.0),
    )

    class DummyContext:
        class DummyLogger:
            def for_step(self, *args, **kwargs):
                return self
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def debug(self, *args, **kwargs): pass
            def error(self, *args, **kwargs): pass
        logger = DummyLogger()

    augmentor = BasicAugmentor()
    input_dto = AugmentationInputDTO(augmentation_config=config, data=split)
    result = augmentor.run(input_dto, DummyContext())

    augmented_train = result.data.train
    assert augmented_train is not None
    assert len(augmented_train.data) == 1

    aug_rec = augmented_train.data[0]
    # 4 original + 2 * 4 copies = 12 samples
    assert aug_rec.data.shape == (12, n_channels, n_times)
    assert len(aug_rec.metadata["labels"]) == 12
    # Per-epoch metadata must match the total augmented length!
    assert len(aug_rec.metadata["trial_ids"]) == 12
    # Essential recording metadata is preserved
    assert aug_rec.metadata["sfreq"] == 250.0
