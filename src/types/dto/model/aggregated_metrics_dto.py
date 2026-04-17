from dataclasses import dataclass, field


@dataclass(frozen=True)
class AggregatedMetricsDTO:
    """
    Data Transfer Object obsahující agregované statistiky z křížové validace.
    """

    model_name: str
    metric_name: str

    # Agregované statistiky
    mean: float
    std: float

    # Detailní výsledky jednotlivých foldů pro případnou vizualizaci/boxplot
    fold_results: dict[int, float] = field(default_factory=dict)
