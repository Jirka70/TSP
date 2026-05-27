from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.pipeline.context.run_context import RunContext


class IModelLoader(ABC):
    @abstractmethod
    def load(self, model_path: Path, run_ctx: RunContext) -> Any:
        """Loads model from file from given path."""
        pass