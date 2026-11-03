import numpy
import pytest
import torch

from tests.unit.eegnet_numerical_fixtures import create_eegnet_model


def test_normalizes_each_epoch_channel_using_its_own_samples():
    model = create_eegnet_model("per_epoch_channel")
    signal = numpy.array([[[1, 2, 3, 4], [10, 20, 30, 40]], [[5, 6, 7, 8], [40, 30, 20, 10]]], dtype=numpy.float32)
    expected = (signal - signal.mean(axis=2, keepdims=True)) / (signal.std(axis=2, keepdims=True) + 1e-6)
    result = model._to_x_tensor(signal)
    numpy.testing.assert_allclose(result.numpy(), expected)
    numpy.testing.assert_array_equal(signal[0, 0], [1, 2, 3, 4])


def test_constant_channels_remain_finite_after_normalization():
    model = create_eegnet_model("per_epoch_channel")
    result = model._to_x_tensor(numpy.ones((3, 2, 4)))
    assert torch.equal(result, torch.zeros_like(result))


def test_rejects_nonfinite_values_created_by_normalization():
    model = create_eegnet_model("per_epoch_channel")
    signal = numpy.full((2, 2, 4), numpy.finfo(numpy.float32).max, dtype=numpy.float32)
    with pytest.raises(ValueError, match="after per-epoch channel normalization"):
        model._to_x_tensor(signal)


@pytest.mark.parametrize("normalization", ["none", "per_epoch_channel"])
@pytest.mark.parametrize("value", [numpy.nan, numpy.inf, 1e100])
def test_rejects_invalid_values_before_normalization(normalization, value):
    model = create_eegnet_model(normalization)
    with pytest.raises(ValueError, match="after float32 conversion"):
        model._to_x_tensor(numpy.full((2, 2, 4), value))


@pytest.mark.parametrize("shape", [(0, 2, 4), (2, 2), (2, 3, 4), (2, 2, 5)])
def test_rejects_invalid_runtime_input_shape(shape):
    model = create_eegnet_model()
    with pytest.raises(ValueError):
        model._to_x_tensor(numpy.ones(shape))
