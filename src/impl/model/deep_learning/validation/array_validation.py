"""Named array operations used at the deep-learning input boundary."""

import numpy
from numpy.typing import NDArray


def is_array_with_dimensions(values: NDArray[numpy.generic], expected_dimensions: int) -> bool:
    return values.ndim == expected_dimensions


def has_only_positive_dimensions(values: NDArray[numpy.generic]) -> bool:
    return all(dimension > 0 for dimension in values.shape)


def has_only_real_numbers(values: NDArray[numpy.generic]) -> bool:
    return values.dtype.kind in "iuf"


def has_only_integer_numbers(values: NDArray[numpy.generic]) -> bool:
    return values.dtype.kind in "iu"


def has_only_finite_numbers(values: NDArray[numpy.generic]) -> bool:
    return bool(numpy.isfinite(values).all())


def are_arrays_equal(first: NDArray[numpy.generic], second: NDArray[numpy.generic]) -> bool:
    return bool(numpy.array_equal(first, second))


def convert_to_array(values: object) -> NDArray[numpy.generic]:
    return numpy.asarray(values)


def copy_as_training_precision(values: NDArray[numpy.generic]) -> NDArray[numpy.generic]:
    with numpy.errstate(over="ignore", invalid="ignore"):
        return values.astype(numpy.float32, copy=True)


def concatenate_recording_arrays(recording_arrays: list[NDArray[numpy.generic]]) -> NDArray[numpy.generic]:
    return numpy.concatenate(recording_arrays, axis=0)
