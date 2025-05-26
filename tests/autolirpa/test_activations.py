import pytest
import torch
import torch.nn as nn
from keras.layers import Activation, LeakyReLU, ReLU

from .conftest import check_layer, check_layer_activation, empirical_check_layer


class ExpModule(nn.Module):
    forward = staticmethod(lambda x: torch.exp(x))


map_keras_2_torch_activation = {
    "relu": torch.nn.ReLU(),
    "softsign": torch.nn.Softsign(),
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "exponential": ExpModule(),  # to correct
    "elu": torch.nn.ELU(),
    "leaky_relu": torch.nn.LeakyReLU(),
    "selu": torch.nn.SELU(),
    "softplus": torch.nn.Softplus(),
}
"""
def _test_backward_activation(keras_layer, torch_layer, input_shape, method):

    # data_format == 'channels_first'
    torch_layer = torch.nn.ReLU()
    check_layer(keras_layer, torch_layer, input_shape, method=method, wo_linearity=True, decimal=6)"
"""


@pytest.mark.parametrize(
    "activation, method",
    [
        ("relu", "forward-ibp"),
        ("relu", "crown-forward-ibp"),
        ("relu", "crown"),
        ("softsign", "forward-ibp"),
        ("softsign", "crown-forward-ibp"),
        ("softsign", "crown"),
        ("sigmoid", "forward-ibp"),
        ("sigmoid", "crown-forward-ibp"),
        ("sigmoid", "crown"),
        ("tanh", "forward-ibp"),
        ("tanh", "crown-forward-ibp"),
        ("tanh", "crown"),
        # ('leaky_relu', "forward-ibp"), ('leaky_relu', "crown-forward-ibp"), ('leaky_relu', "crown"),
    ],
)
def test_backward_activation(activation, method):
    input_dim = 10
    keras_layer = Activation(activation)
    torch_layer = map_keras_2_torch_activation[activation]
    check_layer_activation(keras_layer, torch_layer, input_dim, method=method, decimal=0, finetune=False)
    check_layer_activation(keras_layer, torch_layer, input_dim, method=method, decimal=0, finetune=True)


@pytest.mark.parametrize(
    "activation, method",
    [
        ("relu", "forward-affine"),
        ("relu", "forward-hybrid"),
        ("relu", "crown-forward-ibp"),
        ("relu", "crown"),
        ("softsign", "forward-affine"),
        ("softsign", "forward-hybrid"),
        ("softsign", "crown-forward-ibp"),
        ("softsign", "crown"),
        ("sigmoid", "forward-affine"),
        ("sigmoid", "forward-hybrid"),
        ("sigmoid", "crown-forward-ibp"),
        ("sigmoid", "crown"),
        ("tanh", "forward-affine"),
        ("tanh", "forward-hybrid"),
        ("tanh", "crown-forward-ibp"),
        ("tanh", "crown"),
        ("exponential", "forward-affine"),
        ("exponential", "forward-hybrid"),
        ("exponential", "crown-forward-ibp"),
        ("exponential", "crown"),
        ("elu", "forward-affine"),
        ("elu", "forward-hybrid"),
        ("elu", "crown-forward-ibp"),
        ("elu", "crown"),  # not supported by autoLirpa
        ("leaky_relu", "forward-affine"),
        ("leaky_relu", "forward-hybrid"),
        ("leaky_relu", "crown-forward-ibp"),
        ("leaky_relu", "crown"),
        ("selu", "forward-affine"),
        ("selu", "forward-hybrid"),
        ("selu", "crown-forward-ibp"),
        ("selu", "crown"),  # not supported by autoLirpa
        ("softplus", "forward-affine"),
        ("softplus", "forward-hybrid"),
        ("softplus", "crown-forward-ibp"),
        ("softplus", "crown"),  # not supported by autoLirpa
    ],
)
def test_backward_activation_empirical(activation, method):
    input_dim = (10,)
    keras_layer = Activation(activation)
    empirical_check_layer(keras_layer, input_dim, method=method, decimal=1)
