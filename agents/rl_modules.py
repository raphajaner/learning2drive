import torch.nn as nn
import torch
from agents.utils import layer_orth_init


class ActorMeanNet(nn.Sequential):
    def __init__(self, fc_arch, action_offset=0.0, std=0.01, bias_const=0.0, layer_norm=False):
        """Actor model for mean values of action distribution.

        Args:
            fc_arch (list): List of tuples (in_channels, out_channels) for each layer
            action_offset (float, optional): Offset for action. Defaults to 0.0.
            std (float, optional): Std dev for layer init. Defaults to 0.01.
            bias_const (float, optional): Bias for layer init. Defaults to 0.0.
        """
        network = nn.Sequential()
        network.add_module(f"fc0", nn.LazyLinear(fc_arch[0]))
        if len(fc_arch) > 1:
            if layer_norm:
                network.add_module("ln0", nn.LayerNorm(fc_arch[0]))
            network.add_module(f"tanh0", nn.Tanh())
            for i, (in_channels, out_channels) in enumerate(fc_arch[1:-1]):
                network.add_module(f"fc{i + 1}",
                                   layer_orth_init(nn.Linear(in_channels, out_channels)))
                if layer_norm:
                    network.add_module(f"ln{i + 1}", nn.LayerNorm(out_channels))
                network.add_module(f"tanh{i + 1}", nn.Tanh())
            network.add_module(f"fc{len(fc_arch) - 1}",
                               layer_orth_init(nn.Linear(fc_arch[-1][0], fc_arch[-1][1]), std=std, bias_const=bias_const))
        super().__init__(network)
        self.register_buffer("action_offset", torch.tensor(action_offset, dtype=torch.float32))

    def forward(self, x):
        base = super().forward(x)
        offset = self.action_offset
        return base + offset


class CriticNet(nn.Sequential):
    def __init__(self, fc_arch, std=1.0, bias_const=0.0, layer_norm=False):
        """Critic model for state values.

        Args:
            fc_arch (list): List of tuples (in_channels, out_channels) for each layer
            std (float, optional): Std dev for layer init. Defaults to 1.0.
            bias_const (float, optional): Bias for layer init. Defaults to 0.0.
            layer_norm (bool, optional): Whether to use layer norm. Defaults to True.
        """
        network = nn.Sequential()
        network.add_module(f"fc0", nn.LazyLinear(fc_arch[0]))
        if len(fc_arch) > 1:
            # network.add_module(f"fc0", layer_orth_init(nn.Linear(512, fc_arch[0]))) #nn.LazyLinear(fc_arch[0]))
            if layer_norm:
                network.add_module("ln0", nn.LayerNorm(fc_arch[0]))
            network.add_module(f"tanh0", nn.Tanh())
            for i, (in_channels, out_channels) in enumerate(fc_arch[1:-1]):
                network.add_module(f"fc{i + 1}",
                                   layer_orth_init(nn.Linear(in_channels, out_channels)))
                if layer_norm:
                    network.add_module(f"ln{i + 1}", nn.LayerNorm(out_channels))
                network.add_module(f"tanh{i + 1}", nn.Tanh())
            network.add_module(f"fc_out",
                               # nn.Linear(fc_arch[-1][0], fc_arch[-1][1]))
                               layer_orth_init(nn.Linear(fc_arch[-1][0], fc_arch[-1][1]), std=std, bias_const=bias_const))
        super().__init__(network)
