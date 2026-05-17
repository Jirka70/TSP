from src.impl.model.deep_learning.eegnet_model import EEGNetModel
from src.impl.model.util.network.create_eegnet_network import create_eegnet_network
from src.types.dto.config.model.model_config import EEGNetConfig


def build_eegnet_model(
    config: EEGNetConfig,
    input_shape: tuple[int, int, int],
) -> EEGNetModel:
    network = create_eegnet_network(config, input_shape)

    return EEGNetModel(
        network=network,
        model_name=config.model_name,
        config=config,
    )
