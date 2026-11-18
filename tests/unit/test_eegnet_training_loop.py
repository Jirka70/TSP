"""Verify model selection and validation isolation using real PyTorch networks."""

from unittest.mock import Mock, patch

import numpy
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.impl.model.deep_learning.eegnet.trainer.eegnet_training_loop import EEGNetTrainingLoop
from src.pipeline_logging.pipeline_logger import PipelineLogger
from src.types.dto.model.learning_dataset import LearningDataset
from tests.unit.eegnet_numerical_fixtures import create_eegnet_model


@pytest.mark.parametrize("validation_accuracies", [(0.75, 0.5, 0.25), (0.5, 0.75, 0.75)])
def test_restores_earliest_best_epoch_instead_of_last_epoch(
    validation_accuracies: tuple[float, float, float],
) -> None:
    model = create_eegnet_model()
    signal = numpy.arange(24, dtype=numpy.float32).reshape(3, 2, 4) / 24
    dataset = LearningDataset(signal, numpy.array([3, 7, 3]))
    network_states: list[dict[str, torch.Tensor]] = []
    train_one_epoch = model.train_one_epoch

    def record_training_epoch(
        training_signal: NDArray[numpy.generic],
        training_labels: NDArray[numpy.generic],
    ) -> tuple[float, float]:
        metrics = train_one_epoch(training_signal, training_labels)
        network_states.append(model.get_network_state_dict())
        return metrics

    validation_metrics = [(1.0, accuracy) for accuracy in validation_accuracies]
    with (
        patch.object(model, "train_one_epoch", side_effect=record_training_epoch),
        patch.object(model, "validate", side_effect=validation_metrics) as validation,
    ):
        training_result = EEGNetTrainingLoop.train(
            model=model,
            train_data=dataset,
            validation_data=dataset,
            log=Mock(spec=PipelineLogger),
            epochs=3,
        )

    expected_epoch = validation_accuracies.index(max(validation_accuracies))
    assert validation.call_count == 3
    assert training_result.best_epoch == model.best_epoch == expected_epoch
    assert training_result.best_validation_accuracy == max(validation_accuracies)
    assert model.best_validation_accuracy == max(validation_accuracies)
    assert training_result.best_state_dict is not None
    for parameter_name, parameter in model.get_network_state_dict().items():
        torch.testing.assert_close(parameter, network_states[expected_epoch][parameter_name], rtol=0, atol=0)
        torch.testing.assert_close(
            training_result.best_state_dict[parameter_name],
            network_states[expected_epoch][parameter_name],
            rtol=0,
            atol=0,
        )


def test_validation_preserves_parameters_batch_normalization_state_and_gradients() -> None:
    network = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Flatten(), nn.Linear(8, 2))
    model = create_eegnet_model(network=network)
    signal = numpy.arange(24, dtype=numpy.float32).reshape(3, 2, 4)
    labels = numpy.array([3, 7, 3])
    model.initialize_training(labels)
    model.train_one_epoch(signal, labels)
    expected_network_state = model.get_network_state_dict()
    expected_gradients = _copy_parameter_gradients(model)

    with patch.object(model._optimizer, "step") as update:
        first_metrics = model.validate(signal, labels)
        second_metrics = model.validate(signal, labels)

    update.assert_not_called()
    assert first_metrics == second_metrics
    assert not network.training
    for parameter_name, parameter in model.get_network_state_dict().items():
        torch.testing.assert_close(parameter, expected_network_state[parameter_name], rtol=0, atol=0)
    for gradient, expected_gradient in zip(_copy_parameter_gradients(model), expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


def test_records_each_validation_accuracy_without_fold_metrics() -> None:
    model = create_eegnet_model()
    signal = numpy.arange(24, dtype=numpy.float32).reshape(3, 2, 4)
    labels = numpy.array([3, 7, 3])
    model.initialize_training(labels)
    assert model.history is not None
    assert model.history.validation_metrics == {"validation_accuracy": []}

    first_loss, first_accuracy = model.validate(signal, labels)
    second_loss, second_accuracy = model.validate(signal, labels)

    assert model.history.validation_metrics == {"validation_accuracy": [first_accuracy, second_accuracy]}
    assert model.history.validation_loss == [first_loss, second_loss]


def _copy_parameter_gradients(model: EEGNetModel) -> list[torch.Tensor]:
    gradients: list[torch.Tensor] = []
    for parameter in model._network.parameters():
        if parameter.grad is None:
            raise AssertionError("The training epoch must produce parameter gradients.")
        gradients.append(parameter.grad.detach().clone())
    return gradients


@pytest.mark.parametrize("epoch_count,expected_best_epoch", [(0, None), (2, 1)])
def test_direct_loop_without_validation_preserves_last_epoch_contract(
    epoch_count: int,
    expected_best_epoch: int | None,
) -> None:
    model = create_eegnet_model()
    signal = numpy.ones((3, 2, 4), dtype=numpy.float32)
    dataset = LearningDataset(signal, numpy.array([3, 7, 3]))

    training_result = EEGNetTrainingLoop.train(
        model=model,
        train_data=dataset,
        validation_data=None,
        log=Mock(spec=PipelineLogger),
        epochs=epoch_count,
    )

    assert training_result.best_epoch == model.best_epoch == expected_best_epoch
    assert training_result.best_validation_accuracy is None
    assert training_result.best_state_dict is None
    assert model.history is not None
    assert len(model.history.train_loss) == epoch_count
    assert model.history.validation_loss == []
