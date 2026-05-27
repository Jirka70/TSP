from pathlib import Path
from typing import Any

from src.types.interfaces.model.model_loader import IModelLoader


class EEGNetModelLoader(IModelLoader):
    def load(self, model_path: Path) -> Any:
        pass