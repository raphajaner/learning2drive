import torch
import numpy as np
from torchinfo import summary


def save_model_summary(log_dir, models):
    with open(f'{log_dir}/model_summary.txt', 'w') as file:
        for model in models:
            model_summary = summary(model,
                                    col_names=[
                                        "num_params",
                                        "params_percent",
                                        "kernel_size",
                                        "trainable"
                                    ],
                                    row_settings=("var_names", "depth"),
                                    depth=10,
                                    verbose=0)
            file.write(repr(model_summary) + '\n')


def layer_orth_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer with orthogonal weights and a constant bias.
    Args:
        layer (torch.nn.Module): The layer to be initialized.
        std (float): Gain, optional scaling factor.
        bias_const (float): The constant value of the bias.
    Returns:
        torch.nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_conv_init(layer, nonlinearity='relu'):
    """Initialize a convolutional layer with Kaiming normal weights and zero bias.
    See:
    - https://github.com/pytorch/pytorch/issues/18182
    - https://github.com/pytorch/vision/blob/a75fdd4180683f7953d97ebbcc92d24682690f96/torchvision/models/resnet.py#L160
    - https://github.com/pytorch/pytorch/issues/18182#issuecomment-554376411
    Args:
        layer (torch.nn.Module): The layer to be initialized.
        nonlinearity (str): The nonlinearity used in the layer.
    Returns:
        torch.nn.Module: The initialized layer.
    """
    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=nonlinearity)
    torch.nn.init.zeros_(layer.bias)
    # if m.bias is not None:
    #     fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
    #     bound = 1 / math.sqrt(fan_out)
    #     nn.init.normal_(m.bias, -bound, bound)
    return layer


def layer_linear_init(layer, bias_const=0.0, nonlinearity='relu'):
    # see also https://github.com/pytorch/pytorch/issues/18182
    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=nonlinearity)
    # torch.nn.init.zeros_(layer.bias)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def conv_out_dim(size, conv_arch):
    """Computes the output size of a convolutional network.
    Args:
        size (int): The input size.
        conv_arch (list): The convolutional architecture.
    Returns:
        int: The output size.
    """
    for _, _, kernel_size, stride, padding in conv_arch:
        size = (size - kernel_size + 2 * padding) // stride + 1
    return size ** 2 * conv_arch[-1][1]


def conv_out_size(size, conv_arch):
    """Computes the output size of a convolutional network.
    Args:
        size (int): The input size.
        conv_arch (list): The convolutional architecture.
    Returns:
        int: The output size.
    """
    for _, _, kernel_size, stride, padding in conv_arch:
        size = (size - kernel_size + 2 * padding) // stride + 1
    return size


def pool_out_size(size, kernel_size, stride, padding):
    return (size - kernel_size + 2 * padding) // stride + 1
