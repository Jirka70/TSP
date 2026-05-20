from dataclasses import dataclass, field


@dataclass(frozen=True)
class AggregatedMetricsDTO:
    """Data transfer object holding aggregated cross-validation metrics."""

    model_name: str
    metric_name: str

    # Aggregated statistics.
    mean: float
    std: float

    # Per-fold results for visualization or box plots.
    fold_results: dict[int, float] = field(default_factory=dict)
