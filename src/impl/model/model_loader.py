from typing import Dict, Any
import joblib
from pathlib import Path
import logging

from src.types.interfaces.model.model_loader import IModelLoader


class ModelLoader:
    def __init__(self):
        # Map of suffixes
        self._loaders: Dict[str, IModelLoader] = {
            ".joblib": JoblibModelLoader(),
            # ".pth": PytorchModelLoader(),  <-- Future expansion for other formats
            # ".h5": KerasModelLoader(),
        }

    def load(self, model_path: Path) -> Any:
        suffix = model_path.suffix.lower()
        loader = self._loaders.get(suffix)

        if not loader:
            raise ValueError(f"Unsupported model format: {suffix}. "
                             f"Available formats: {list(self._loaders.keys())}")

        return loader.load(model_path)


class JoblibModelLoader(IModelLoader):
    def __init__(self):
        self._log = logging.getLogger(__name__)

    def load(self, model_path: Path) -> Any:
        if not model_path.exists():
            raise FileNotFoundError(f"Model at path {model_path} does not exist.")

        self._log.info(f"Loading model (joblib) from: {model_path}")
        loaded_content = joblib.load(model_path)

        if isinstance(loaded_content, dict) and "pipeline" in loaded_content:
            self._log.info("Detected metadata dictionary, extracting internal model object.")
            return loaded_content["pipeline"]
        raise TypeError(f"Unsupported model format: {model_path}")