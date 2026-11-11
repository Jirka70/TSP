"""Aggregate batch means without changing the gradients used for training."""

from dataclasses import dataclass

import torch


@dataclass
class EpochMetrics:
    weighted_loss_sum: float = 0.0
    total_batch_weight: float = 0.0
    correct_sample_count: int = 0
    sample_count: int = 0

    def record_batch(
        self,
        class_scores: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
        configured_batch_size: int,
    ) -> None:
        batch_sample_count = labels.size(0)
        batch_weight = batch_sample_count / configured_batch_size
        self.weighted_loss_sum += loss.item() * batch_weight
        self.total_batch_weight += batch_weight
        predicted_indices = torch.argmax(class_scores, dim=1)
        self.correct_sample_count += int((predicted_indices == labels).sum().item())
        self.sample_count += batch_sample_count

    def averages(self) -> tuple[float, float]:
        if self.sample_count == 0:
            raise ValueError("Cannot compute epoch metrics without samples.")
        return (
            self.weighted_loss_sum / self.total_batch_weight,
            self.correct_sample_count / self.sample_count,
        )
