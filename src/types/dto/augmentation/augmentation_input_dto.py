from dataclasses import dataclass

from src.types.dto.training.prepared_samples_dto import PreparedSamplesDTO


@dataclass(frozen=True)
class AugmentationInputDTO:
    samples: PreparedSamplesDTO
    enabled: bool

