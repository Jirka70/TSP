from braindecode.models import EEGNet


def create_eegnet_network(config):
    return EEGNet(
        n_chans=config.n_channels,
        n_outputs=config.n_classes,
        n_times=config.n_times,
        final_conv_length="auto",
        drop_prob=config.dropout,
        kernel_length=config.kernel_length,
        F1=config.f1,
        D=config.d,
        F2=config.f2,
    )
