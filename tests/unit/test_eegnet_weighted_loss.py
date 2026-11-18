from unittest.mock import patch

import numpy
import pytest
import torch
from torch import nn

from tests.unit.eegnet_numerical_fixtures import create_eegnet_model


def _create_model_with_fixed_scores():
    network = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))
    with torch.no_grad():
        network[1].weight[0].fill_(0.1)
        network[1].weight[1].fill_(-0.1)
        network[1].bias.zero_()
    model = create_eegnet_model(network=network)
    model.initialize_training(numpy.array([0, 1]))
    return model


@pytest.mark.parametrize("sample_count", [1, 2, 3, 5])
@pytest.mark.parametrize("operation", ["train_one_epoch", "validate", "evaluate"])
def test_reported_loss_matches_mean_over_samples_with_partial_batches(sample_count, operation):
    model = _create_model_with_fixed_scores()
    signal = numpy.repeat(numpy.arange(1, sample_count + 1, dtype=numpy.float32), 8).reshape(sample_count, 2, 4)
    labels = numpy.arange(sample_count) % 2
    with torch.no_grad():
        scores = model._network(torch.tensor(signal))
        expected_loss = nn.functional.cross_entropy(scores, torch.tensor(labels)).item()
        expected_accuracy = (scores.argmax(dim=1) == torch.tensor(labels)).float().mean().item()

    # Freeze parameter updates so every sample is measured against the same model.
    # The real forward, loss and backward computations still run for training.
    with patch.object(model._optimizer, "step"):
        actual_loss, actual_accuracy = getattr(model, operation)(signal, labels)

    assert actual_loss == pytest.approx(expected_loss, rel=1e-6)
    assert actual_accuracy == pytest.approx(expected_accuracy)


def test_partial_batch_weight_does_not_scale_gradients():
    model = _create_model_with_fixed_scores()
    signal = numpy.ones((3, 2, 4), dtype=numpy.float32)
    labels = numpy.array([0, 1, 1])
    parameter = next(model._network.parameters())
    scores = model._network(torch.tensor(signal[-1:]))
    last_batch_mean_loss = nn.functional.cross_entropy(scores, torch.tensor(labels[-1:]))
    expected_gradient = torch.autograd.grad(last_batch_mean_loss, parameter)[0]
    training_loader = model._create_loader(signal, labels, shuffle=False)

    with patch.object(model._optimizer, "step"):
        model._run_training_loader(training_loader)

    torch.testing.assert_close(parameter.grad, expected_gradient)
