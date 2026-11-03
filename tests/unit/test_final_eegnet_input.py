from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.impl.model.deep_learning.eegnet.trainer.final_eegnet_trainer import FinalEEGNetTrainer
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.config.model.training_config import TrainingConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.load.recording import RecordingDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from src.types.dto.split.dataset_split_dto import FoldDTO


def _dataset(value, count=2):
    return EpochPreprocessedDTO(data=[RecordingDTO(
        data=np.full((count, 2, 32), value, dtype=np.float32),
        dataset_name="synthetic",
        subject_id=1,
        session_id="0",
        run_id="0",
        metadata={"labels": np.arange(count) % 2},
    )])


@pytest.fixture
def training_input():
    config = EEGNetConfig.model_construct(
        model_name="eegnet",
        training=TrainingConfig(
            epochs=1, batch_size=2, learning_rate=0.001,
            optimizer="adam", random_state=None,
        ),
    )
    return FinalTrainingInputDTO(
        config=config,
        training_data=_dataset(99, count=8),
        folds=[FoldDTO(fold_idx=0, train_data=_dataset(1), test_data=_dataset(3))],
        validation_data=_dataset(2),
    )


@pytest.mark.parametrize("count", [0, 2])
def test_run_rejects_wrong_fold_count_before_creating_model(training_input, count):
    factory = Mock()
    trainer = FinalEEGNetTrainer(model_factory=factory)
    invalid_input = replace(training_input, folds=training_input.folds * count)

    with pytest.raises(ValueError, match="exactly one fold"):
        trainer.run(invalid_input, Mock())

    factory.create.assert_not_called()


@pytest.mark.parametrize("name", ["train", "val", "test"])
@pytest.mark.parametrize("missing", [None, EpochPreprocessedDTO(data=[]), _dataset(0, count=0)])
def test_rejects_missing_or_empty_sets(training_input, name, missing):
    fold = training_input.folds[0]
    if name == "val":
        invalid_input = replace(training_input, validation_data=missing)
    else:
        invalid_input = replace(
            training_input,
            folds=[replace(fold, **{f"{name}_data": missing})],
        )

    with pytest.raises(ValueError, match=name):
        FinalEEGNetTrainer()._to_dl_input(invalid_input)


def test_run_uses_fold_train_and_validation_without_training_on_test(training_input):
    factory = Mock()
    trainer = FinalEEGNetTrainer(model_factory=factory)
    dl_input = trainer._to_dl_input(training_input)
    assert dl_input.train is training_input.folds[0].train_data
    assert dl_input.val is training_input.validation_data
    assert dl_input.test is training_input.folds[0].test_data

    run_ctx = SimpleNamespace(logger=Mock(), run_id="test")
    with patch(
        "src.impl.model.deep_learning.eegnet.trainer.final_eegnet_trainer.EEGNetTrainingLoop.train",
        return_value=SimpleNamespace(best_epoch=1, best_validation_accuracy=0.5),
    ) as train:
        result = trainer.run(training_input, run_ctx)

    np.testing.assert_array_equal(train.call_args.kwargs["train_data"].x, dl_input.train.data[0].data)
    np.testing.assert_array_equal(train.call_args.kwargs["validation_data"].x, dl_input.val.data[0].data)
    assert result.data.trained_model.metadata["training_data_source"] == "fold.train_data"
    assert factory.create.call_args.kwargs["input_shape"] == (2, 2, 32)
