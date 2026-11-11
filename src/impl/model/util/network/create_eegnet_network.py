"""Construct the EEGNet architecture used by training and checkpoint loading."""

from braindecode.models import EEGNet
from torch import nn

from src.types.dto.config.model.model_config import EEGNetConfig


def create_eegnet_network(config: EEGNetConfig, shape: tuple[int, int, int]) -> nn.Module:
    network = EEGNet(
        n_chans=shape[1],
        n_outputs=config.n_classes,
        n_times=shape[2],
        final_conv_length="auto",
        drop_prob=config.dropout,
        kernel_length=config.kernel_length,
        F1=config.temporal_filters,
        D=config.depth_multiplier,
        F2=config.pointwise_filters,
        batch_norm_momentum=0.2,
    )
    if not isinstance(network, nn.Module):
        raise TypeError("The EEGNet constructor must return a PyTorch module.")
    return network
