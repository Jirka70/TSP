from unittest.mock import patch

import numpy
import pytest
import torch
from numpy.typing import NDArray
from torch import nn

from tests.unit.eegnet_numerical_fixtures import create_eegnet_model


class NonfiniteLoss(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, scores, labels):
        return scores.sum() * 0 + self.value


@pytest.fixture
def training_arrays():
    return numpy.ones((2, 2, 4), dtype=numpy.float32), numpy.array([3, 7])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_training_loss_before_backward_or_optimizer_update(
    training_arrays: tuple[NDArray[numpy.float32], NDArray[numpy.int64]],
    value: float,
) -> None:
    signal, labels = training_arrays
    model = create_eegnet_model()
    model.initialize_training(labels)
    model._loss_function = NonfiniteLoss(value)
    with patch.object(model._optimizer, "step") as update:
        with pytest.raises(FloatingPointError, match="Training epoch 1, batch 1: loss"):
            model.train_one_epoch(signal, labels)
    update.assert_not_called()
    assert all(parameter.grad is None for parameter in model._network.parameters())
    assert model.history.train_loss == []


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_validation_loss_without_recording_it(
    training_arrays: tuple[NDArray[numpy.float32], NDArray[numpy.int64]],
    value: float,
) -> None:
    signal, labels = training_arrays
    model = create_eegnet_model()
    model.initialize_training(labels)
    model._loss_function = NonfiniteLoss(value)
    with pytest.raises(FloatingPointError, match="Evaluation after epoch 0, batch 1: loss"):
        model.validate(signal, labels)
    assert model.history.validation_loss == []


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_gradients_before_optimizer_update(training_arrays, value):
    signal, labels = training_arrays
    model = create_eegnet_model()
    model.initialize_training(labels)
    parameter = next(model._network.parameters())
    original_parameter = parameter.detach().clone()
    parameter.register_hook(lambda gradient: torch.full_like(gradient, value))
    with patch.object(model._optimizer, "step") as update:
        with pytest.raises(FloatingPointError, match="Training epoch 1, batch 1: gradient"):
            model.train_one_epoch(signal, labels)
    update.assert_not_called()
    assert torch.equal(parameter.detach(), original_parameter)
    assert model.history.train_loss == []


def test_rejects_corrupt_parameters_after_optimizer_update(training_arrays):
    signal, labels = training_arrays
    model = create_eegnet_model()
    model.initialize_training(labels)

    def corrupt_parameter():
        with torch.no_grad():
            next(model._network.parameters()).fill_(float("inf"))

    with patch.object(model._optimizer, "step", side_effect=corrupt_parameter):
        with pytest.raises(FloatingPointError, match="parameter.*after update"):
            model.train_one_epoch(signal, labels)
    assert model.history.train_loss == []


@pytest.mark.parametrize("operation", ["train_one_epoch", "validate", "predict_class_probability"])
def test_rejects_wrong_class_score_shape(training_arrays, operation):
    signal, labels = training_arrays
    model = create_eegnet_model(network=nn.Sequential(nn.Flatten(), nn.Linear(8, 3)))
    model.initialize_training(labels)
    arguments = (signal,) if operation == "predict_class_probability" else (signal, labels)
    with pytest.raises(ValueError, match="expected class score shape"):
        getattr(model, operation)(*arguments)


@pytest.mark.parametrize("operation", ["train_one_epoch", "validate", "predict_class_probability"])
def test_rejects_nonfinite_class_scores(training_arrays, operation):
    signal, labels = training_arrays
    model = create_eegnet_model()
    model.initialize_training(labels)
    arguments = (signal,) if operation == "predict_class_probability" else (signal, labels)
    with patch.object(model._network, "forward", return_value=torch.full((2, 2), float("nan"))):
        with pytest.raises(FloatingPointError, match="class scores contain"):
            getattr(model, operation)(*arguments)


@pytest.mark.parametrize(
    "probabilities,message",
    [
        ([[float("nan"), 0], [0, 1]], "NaN or infinity"),
        ([[float("inf"), 0], [0, 1]], "NaN or infinity"),
        ([[-0.1, 1.1], [0, 1]], "between zero and one"),
        ([[0.2, 0.2], [0, 1]], "sum to one"),
        ([[1, 0]], "expected probability shape"),
    ],
)
def test_prediction_rejects_invalid_probabilities(training_arrays, probabilities, message):
    signal, labels = training_arrays
    model = create_eegnet_model()
    model.initialize_training(labels)
    with patch("torch.softmax", return_value=torch.tensor(probabilities, dtype=torch.float32)):
        with pytest.raises((ValueError, FloatingPointError), match=message):
            model.predict_class_probability(signal)


def test_valid_training_validation_and_prediction_keep_finite_results(training_arrays):
    signal, labels = training_arrays
    model = create_eegnet_model("per_epoch_channel")
    model.initialize_training(labels)
    training_loss, _ = model.train_one_epoch(signal, labels)
    validation_loss, _ = model.validate(signal, labels)
    probabilities = model.predict_class_probability(signal[:1])
    predicted_labels = model.predict(signal[:1])
    assert numpy.isfinite([training_loss, validation_loss]).all()
    assert probabilities.shape == (1, 2)
    assert numpy.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    numpy.testing.assert_allclose(probabilities.sum(axis=1), [1], rtol=1e-5, atol=1e-6)
    assert predicted_labels.shape == (1,)
    assert set(predicted_labels) <= {3, 7}
