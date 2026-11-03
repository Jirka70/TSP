from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy
import pytest

from src.impl.model.deep_learning.eegnet.trainer.final_eegnet_trainer import FinalEEGNetTrainer
from src.types.dto.config.model.model_config import EEGNetConfig, SklearnModelConfig
from src.types.dto.config.model.training_config import TrainingConfig
from src.types.dto.epoch_preprocessing.epoch_preprocessed_dto import EpochPreprocessedDTO
from src.types.dto.model.final_training_input_dto import FinalTrainingInputDTO
from tests.unit.deep_learning_fixtures import create_dataset


@pytest.fixture
def training_input():
    config = EEGNetConfig.model_construct(
        model_name="eegnet",
        n_classes=2,
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            optimizer="adam",
            random_state=None,
        ),
    )
    return FinalTrainingInputDTO(
        config=config,
        training_data=create_dataset(1),
        validation_data=create_dataset(2),
    )


@pytest.mark.parametrize("field_name,dataset_name", [("training_data", "train"), ("validation_data", "val")])
@pytest.mark.parametrize("missing", [None, EpochPreprocessedDTO(data=[]), create_dataset(0, count=0)])
def test_rejects_missing_or_empty_sets_before_creating_model(training_input, field_name, dataset_name, missing):
    factory = Mock()
    invalid_input = replace(training_input, **{field_name: missing})

    with pytest.raises(ValueError, match=dataset_name):
        FinalEEGNetTrainer(factory).run(invalid_input, Mock())

    factory.create.assert_not_called()


def test_rejects_non_eegnet_configuration_before_creating_model(training_input):
    factory = Mock()
    invalid_input = replace(training_input, config=SklearnModelConfig.model_construct())
    with pytest.raises(ValueError, match="requires EEGNetConfig"):
        FinalEEGNetTrainer(factory).run(invalid_input, Mock())
    factory.create.assert_not_called()


def test_run_uses_direct_training_and_validation_data_without_folds_or_test(training_input):
    factory = Mock()
    trainer = FinalEEGNetTrainer(model_factory=factory)
    run_context = SimpleNamespace(logger=Mock(), run_id="test")
    with patch(
        "src.impl.model.deep_learning.eegnet.trainer.final_eegnet_trainer.EEGNetTrainingLoop.train",
        return_value=SimpleNamespace(best_epoch=1, best_validation_accuracy=0.5),
    ) as train:
        result = trainer.run(training_input, run_context)

    numpy.testing.assert_array_equal(train.call_args.kwargs["train_data"].x, training_input.training_data.data[0].data)
    numpy.testing.assert_array_equal(
        train.call_args.kwargs["validation_data"].x,
        training_input.validation_data.data[0].data,
    )
    assert set(train.call_args.kwargs) == {"model", "train_data", "validation_data", "epochs", "log"}
    assert result.data.trained_model.metadata["training_data_source"] == "training_data"
    assert "n_folds" not in result.data.trained_model.metadata
    assert factory.create.call_args.kwargs["input_shape"] == (2, 2, 32)
