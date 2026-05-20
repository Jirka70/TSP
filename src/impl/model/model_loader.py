from typing import Dict, Any
import joblib
from pathlib import Path
import logging

from src.types.interfaces.model.model_loader import IModelLoader


class ModelLoader:
    """
    Facade class that delegates model loading to the appropriate concrete loader
    based on the file extension.
    """

    def __init__(self):
        # Registry mapping file extensions to their corresponding concrete loaders
        self._loaders: Dict[str, IModelLoader] = {
            ".joblib": JoblibModelLoader(),
            # ".pth": PytorchModelLoader(),  # Future expansion for PyTorch models
            # ".h5": KerasModelLoader(),     # Future expansion for Keras models
        }

    def load(self, model_path: Path) -> Any:
        """
        Resolves the appropriate loader for the given file extension and loads the model.

        Args:
            model_path (Path): Path to the serialized model file.

        Returns:
            Any: The deserialized model instance or its internal state.

        Raises:
            ValueError: If no loader is registered for the given file extension.
        """
        suffix : str = model_path.suffix.lower()
        loader : IModelLoader | None = self._loaders.get(suffix)

        if not loader:
            raise ValueError(f"Unsupported model format: {suffix}. "
                             f"Available formats: {list(self._loaders.keys())}")

        return loader.load(model_path)


class JoblibModelLoader(IModelLoader):
    """
    Concrete implementation of IModelLoader for Scikit-learn models serialized via joblib.
    """
    def __init__(self):
        self._log = logging.getLogger(__name__)

    def load(self, model_path: Path) -> Any:
        """
        Loads a Scikit-learn model or state dictionary from a .joblib file.

        Args:
            model_path (Path): Path to the .joblib file.

        Returns:
            Any: The loaded model, pipeline, or state dictionary.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            TypeError: If the loaded content is empty or invalid.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model at path {model_path} does not exist.")

        self._log.info(f"Loading model (joblib) from: {model_path}")
        loaded_content = joblib.load(model_path)

        # Handle case where the model is wrapped in a metadata dictionary
        if isinstance(loaded_content, dict) and "pipeline" in loaded_content:
            self._log.info("Detected metadata dictionary, extracting internal model object.")
            return loaded_content["pipeline"]

        raise TypeError(f"Unsupported model format: {model_path}")