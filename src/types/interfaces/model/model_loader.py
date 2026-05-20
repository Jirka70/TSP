from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class IModelLoader(ABC):
    @abstractmethod
    def load(self, model_path: Path) -> Any:
        """Loads model from file from given path."""
        pass