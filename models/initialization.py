import torch
from torch import nn

def get_activation_name(activation):
    """
    Given a string or a torch.nn.modules.activation, return the name of the activation function.

    Parameters:
        activation: str or torch.nn.modules.activation
            Activation function or its name.

    Returns:
        str: Name of the activation function.
    """
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid"  # Note: Softmax mapped to sigmoid, check if it's intended
    }

    for k, v in mapper.items():
        if isinstance(activation, k):
            return v

    raise ValueError("Unknown given activation type: {}".format(activation))


def get_gain(activation):
    """
    Given an object of torch.nn.modules.activation or an activation name, return the correct gain.

    Parameters:
        activation: torch.nn.modules.activation or str
            Activation function or its name.

    Returns:
        float: The calculated gain value.
    """
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(layer, activation="relu"):
    """
    Initialize a layer.

    Args:
        layer : Parameters to initialize.
        activation (torch.nn.modules.activation or str, optional): Activation function that
            will be used on the layer.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
    """
    Initialize weights of a module based on its type.

    Args:
        module: torch.nn.Module
            The module to initialize weights for.
    """
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)

