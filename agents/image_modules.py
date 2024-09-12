import torch.nn as nn
from agents.utils import *


class ImageNet(nn.Sequential):
    def __init__(self, conv_arch, in_dim, out_dim=512, layer_norm=False, pool_first=False, pool_last=False):
        """ Initialize the network
        Args:
            conv_arch: list of tuples (in, out, kernel, stride, padding) for conv layers
            out_dim: dimension of the output feature vector
        """
        self.n_filter = conv_arch[0][0]
        # build the network given the architecture
        network = nn.Sequential()
        size = init_size = in_dim
        for i, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(conv_arch):
            network.add_module(
                f"conv{i}",
                layer_orth_init(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                ),
            )
            size = conv_out_size(size=size, conv_arch=[(in_channels, out_channels, kernel_size, stride, padding)])
            if pool_first and i == 0:
                # size = init_size // 2
                network.add_module("pool", nn.MaxPool2d(kernel_size=3, stride=2))
                size = pool_out_size(size, kernel_size=3, stride=2, padding=0)

            network.add_module(f"relu{i}", nn.ReLU())

            if layer_norm:
                network.add_module(f'ln{i}', nn.LayerNorm([out_channels, size, size]))

        if pool_last:
            network.add_module("pool", nn.MaxPool2d(kernel_size=3, stride=2))
            size = pool_out_size(size, kernel_size=3, stride=2, padding=0)

        network.add_module("flatten", nn.Flatten())
        flat_linear_dim = size ** 2 * out_channels  # conv_out_dim(init_size, conv_arch)
        network.add_module("fc", layer_orth_init(nn.Linear(flat_linear_dim, out_dim)))
        network.add_module("tanh_flat", nn.Tanh())
        if layer_norm:
            network.add_module("ln_flat", nn.LayerNorm(out_dim))
        super().__init__(network)


def make_image_module(config, in_channels=3):
    return ImageNet(config.rl.image.conv_arch, config.env.sensors.size_output_image, config.rl.hidden_out_dim,
                    config.rl.layer_norm, config.rl.image.pool_first, config.rl.image.pool_last)
