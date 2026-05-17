from src.impl.model.deep_learning.deep_learning_model_registry import DeepLearningModelRegistry
from src.impl.model.deep_learning.eegnet_builder import build_eegnet_model
from src.impl.model.deep_learning.eegnet_model import EEGNetModel
from src.types.dto.config.model.model_config import EEGNetConfig


def create_default_deep_learning_model_registry() -> DeepLearningModelRegistry:
    registry = DeepLearningModelRegistry()
    registry.register("eegnet", build_eegnet_model)
    return registry


class DeepLearningModelFactory:
    def __init__(self, registry: DeepLearningModelRegistry | None = None) -> None:
        self._registry = registry or create_default_deep_learning_model_registry()

    def create(
        self,
        config: EEGNetConfig,
        input_shape: tuple[int, int, int],
    ) -> EEGNetModel:
        return self._registry.create(config, input_shape)
