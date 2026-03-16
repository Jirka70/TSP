from abc import ABC, abstractmethod
from pathlib import Path

from src.types.dto.model.trained_model_dto import TrainedModelDTO


class IModelSerializer(ABC):

    @abstractmethod
    def supports(self, model_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        trained_model: TrainedModelDTO,
        output_path: Path,
    ) -> None:
        raise NotImplementedError
