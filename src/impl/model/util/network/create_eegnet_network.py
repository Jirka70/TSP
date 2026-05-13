from braindecode.models import EEGNet


def create_eegnet_network(config, shape: tuple[int, int, int]):
    return EEGNet(n_chans=shape[1], n_outputs=config.n_classes, n_times=shape[2], final_conv_length="auto", drop_prob=config.dropout, kernel_length=config.kernel_length, F1=config.f1, D=config.d, F2=config.f2, batch_norm_momentum=0.2)
