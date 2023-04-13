import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, Conv2D, Dense, Permute

from decomon.models.utils import split_activation


@pytest.mark.parametrize(
    "layer_class, layer_kwargs, input_shape_wo_batchsize",
    [
        (Dense, {"units": 3, "activation": "relu"}, (1,)),
        (Conv2D, {"filters": 2, "kernel_size": (3, 3), "activation": "relu"}, (64, 64, 3)),
    ],
)
def test_split_activation_do_split(layer_class, layer_kwargs, input_shape_wo_batchsize, use_bias):
    # init layer
    layer = layer_class(use_bias=use_bias, **layer_kwargs)
    # init input_shape and weights
    input_shape = (None,) + input_shape_wo_batchsize
    input_tensor = Input(input_shape)
    layer(input_tensor)
    # split
    layers = split_activation(layer)
    # check layer split
    assert len(layers) == 2
    layer_wo_activation, activation_layer = layers
    assert isinstance(layer_wo_activation, layer.__class__)
    assert layer_wo_activation.get_config()["activation"] == "linear"
    assert isinstance(activation_layer, Activation)
    assert activation_layer.get_config()["activation"] == layer_kwargs["activation"]
    # check names starts with with original name + "_"
    assert layer_wo_activation.name.startswith(f"{layer.name}_")
    assert activation_layer.name.startswith(f"{layer.name}_")
    # check already built
    assert layer_wo_activation.built
    assert activation_layer.built
    # check same outputs
    input_shape_with_batch_size = (5,) + input_shape_wo_batchsize
    flatten_dim = np.prod(input_shape_with_batch_size)
    inputs_np = np.linspace(-1, 1, flatten_dim).reshape(input_shape_with_batch_size)
    output_np_ref = layer(inputs_np).numpy()
    output_np_new = activation_layer(layer_wo_activation(inputs_np)).numpy()
    assert_almost_equal(output_np_new, output_np_ref)
    # check same trainable weights
    for i in range(len(layer._trainable_weights)):
        assert layer._trainable_weights[i] is layer_wo_activation._trainable_weights[i]


@pytest.mark.parametrize(
    "layer_class, layer_kwargs",
    [
        (Dense, {"units": 3}),
        (Activation, {"activation": "relu"}),
        (Permute, {"dims": (1, 2, 3)}),
    ],
)
def test_split_activation_do_nothing(layer_class, layer_kwargs):
    layer = layer_class(**layer_kwargs)
    layers = split_activation(layer)
    assert len(layers) == 1
    assert layers[0] == layer


def test_split_activation_uninitialized_layer_ko():
    layer = Dense(3, activation="relu")
    with pytest.raises(RuntimeError):
        layers = split_activation(layer)
