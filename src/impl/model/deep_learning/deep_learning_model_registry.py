from collections.abc import Callable
from dataclasses import dataclass, field

from src.impl.model.deep_learning.eegnet_model import EEGNetModel
from src.types.dto.config.model.model_config import EEGNetConfig


DeepLearningModelBuilder = Callable[[EEGNetConfig, tuple[int, int, int]], EEGNetModel]


@dataclass
class DeepLearningModelRegistry:
    _builders: dict[str, DeepLearningModelBuilder] = field(default_factory=dict)

    def register(self, model_name: str, builder: DeepLearningModelBuilder) -> None:
        if model_name in self._builders:
            raise ValueError(f"Deep learning model already registered: {model_name}")

        self._builders[model_name] = builder

    def create(self, config: EEGNetConfig, input_shape: tuple[int, int, int]) -> EEGNetModel:
        builder = self._builders.get(config.model_name)

        if builder is None:
            available_models = ", ".join(sorted(self._builders))
            raise ValueError(
                f"Unsupported deep learning model: {config.model_name}. "
                f"Available models: {available_models}"
            )

        return builder(config, input_shape)
