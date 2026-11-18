"""Numerical checks at training and prediction boundaries."""

import torch
from torch import nn


def has_only_finite_tensor_values(values: torch.Tensor) -> bool:
    return bool(torch.isfinite(values).all().item())


def has_probability_values_in_unit_interval(values: torch.Tensor) -> bool:
    return bool(((values >= 0) & (values <= 1)).all().item())


def has_probability_rows_summing_to_one(probabilities: torch.Tensor) -> bool:
    row_sums = probabilities.sum(dim=1)
    expected_sums = torch.ones_like(row_sums)
    return bool(torch.allclose(row_sums, expected_sums, rtol=1e-5, atol=1e-6))


def validate_class_scores(
    scores: torch.Tensor,
    sample_count: int,
    class_count: int,
    context: str,
) -> None:
    if tuple(scores.shape) != (sample_count, class_count):
        raise ValueError(
            f"{context}: expected class score shape {(sample_count, class_count)}, got {tuple(scores.shape)}."
        )
    if not has_only_finite_tensor_values(scores):
        raise FloatingPointError(f"{context}: class scores contain NaN or infinity.")


def validate_finite_loss(loss: torch.Tensor, context: str) -> None:
    if loss.ndim != 0:
        raise ValueError(f"{context}: loss must be a scalar.")
    if not has_only_finite_tensor_values(loss):
        raise FloatingPointError(f"{context}: loss contains NaN or infinity.")


def validate_finite_gradients(network: nn.Module, context: str) -> None:
    for parameter_name, parameter in network.named_parameters():
        if parameter.grad is None:
            continue
        if not has_only_finite_tensor_values(parameter.grad):
            raise FloatingPointError(f"{context}: gradient of '{parameter_name}' contains NaN or infinity.")


def validate_finite_parameters(network: nn.Module, context: str) -> None:
    for parameter_name, parameter in network.named_parameters():
        if not has_only_finite_tensor_values(parameter):
            raise FloatingPointError(f"{context}: parameter '{parameter_name}' contains NaN or infinity after update.")


def validate_class_probabilities(
    probabilities: torch.Tensor,
    sample_count: int,
    class_count: int,
    context: str,
) -> None:
    if tuple(probabilities.shape) != (sample_count, class_count):
        raise ValueError(f"{context}: expected probability shape {(sample_count, class_count)}.")
    if not has_only_finite_tensor_values(probabilities):
        raise FloatingPointError(f"{context}: probabilities contain NaN or infinity.")
    if not has_probability_values_in_unit_interval(probabilities):
        raise ValueError(f"{context}: probabilities must be between zero and one.")
    if not has_probability_rows_summing_to_one(probabilities):
        raise ValueError(f"{context}: each probability row must sum to one.")
