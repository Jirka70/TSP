"""Small real PyTorch model used to test numerical failure boundaries."""

from torch import nn

from src.impl.model.deep_learning.eegnet.model.eegnet_model import EEGNetModel
from src.types.dto.config.model.model_config import EEGNetConfig
from src.types.dto.config.model.training_config import TrainingConfig


def create_eegnet_model(
    normalization: str = "none",
    network: nn.Module | None = None,
) -> EEGNetModel:
    configuration = EEGNetConfig.model_construct(
        model_name="eegnet",
        n_classes=2,
        input_normalization=normalization,
        training=TrainingConfig(epochs=2, batch_size=2, learning_rate=0.001, optimizer="sgd", random_state=None),
    )
    if network is None:
        network = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))
    model = EEGNetModel(network, "eegnet", configuration, (2, 2, 4))
    model._device = "cpu"
    model._network.to("cpu")
    return model
