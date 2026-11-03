"""Named array operations used at the deep-learning input boundary."""

import numpy


def is_array_with_dimensions(values: numpy.ndarray, expected_dimensions: int) -> bool:
    return values.ndim == expected_dimensions


def has_only_positive_dimensions(values: numpy.ndarray) -> bool:
    return all(dimension > 0 for dimension in values.shape)


def has_only_real_numbers(values: numpy.ndarray) -> bool:
    return values.dtype.kind in "iuf"


def has_only_integer_numbers(values: numpy.ndarray) -> bool:
    return values.dtype.kind in "iu"


def has_only_finite_numbers(values: numpy.ndarray) -> bool:
    return bool(numpy.isfinite(values).all())


def are_arrays_equal(first: numpy.ndarray, second: numpy.ndarray) -> bool:
    return bool(numpy.array_equal(first, second))


def are_numbers_close_to_each_other(first: float, second: float) -> bool:
    return bool(numpy.isclose(first, second, rtol=1e-9, atol=1e-9))


def convert_to_array(values: object) -> numpy.ndarray:
    return numpy.asarray(values)


def copy_as_training_precision(values: numpy.ndarray) -> numpy.ndarray:
    with numpy.errstate(over="ignore", invalid="ignore"):
        return values.astype(numpy.float32, copy=True)


def concatenate_recording_arrays(recording_arrays: list[numpy.ndarray]) -> numpy.ndarray:
    return numpy.concatenate(recording_arrays, axis=0)
