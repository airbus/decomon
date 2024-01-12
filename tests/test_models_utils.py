import keras
import keras.ops as K
import numpy as np
import pytest
from keras import Input
from keras.layers import Activation, Conv2D, Dense, Layer, Permute, PReLU
from numpy.testing import assert_almost_equal

from decomon.keras_utils import get_weight_index_from_name
from decomon.models.utils import preprocess_layer, split_activation


@pytest.mark.parametrize(
    "layer_class, layer_kwargs, input_shape_wo_batchsize, embedded_activation_layer_class",
    [
        (Dense, {"units": 3, "activation": "relu"}, (1,), None),
        (Dense, {"units": 3}, (1,), PReLU),
        (Conv2D, {"filters": 2, "kernel_size": (3, 3), "activation": "relu"}, (64, 64, 3), None),
    ],
)
def test_split_activation_do_split(
    layer_class, layer_kwargs, input_shape_wo_batchsize, embedded_activation_layer_class, use_bias
):
    # init layer
    if embedded_activation_layer_class is not None:
        layer = layer_class(use_bias=use_bias, activation=embedded_activation_layer_class(), **layer_kwargs)
    else:
        layer = layer_class(use_bias=use_bias, **layer_kwargs)
    # init input_shape and weights
    input_shape = input_shape_wo_batchsize
    input_tensor = Input(input_shape)
    layer(input_tensor)
    # split
    layers = split_activation(layer)
    # check layer split
    assert len(layers) == 2
    layer_wo_activation, activation_layer = layers
    assert isinstance(layer_wo_activation, layer.__class__)
    assert layer_wo_activation.get_config()["activation"] == "linear"
    if isinstance(layer.activation, Layer):
        assert activation_layer == layer.activation
    else:
        assert isinstance(activation_layer, Activation)
        assert activation_layer.get_config()["activation"] == layer_kwargs["activation"]
    # check names starts with original name + "_"
    assert layer_wo_activation.name.startswith(f"{layer.name}_")
    assert activation_layer.name.startswith(f"{layer.name}_")
    # check already built
    assert layer_wo_activation.built
    assert activation_layer.built
    # check same outputs
    input_shape_with_batch_size = (5,) + input_shape_wo_batchsize
    flatten_dim = np.prod(input_shape_with_batch_size)
    inputs_np = np.linspace(-1, 1, flatten_dim).reshape(input_shape_with_batch_size)
    output_np_ref = K.convert_to_numpy(layer(inputs_np))
    output_np_new = K.convert_to_numpy(activation_layer(layer_wo_activation(inputs_np)))
    assert_almost_equal(output_np_new, output_np_ref)
    # check same weights (really same objects)
    for i in range(len(layer_wo_activation.weights)):
        assert layer.weights[i] is layer_wo_activation.weights[i]


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
    with pytest.raises(ValueError):
        layers = split_activation(layer)


@pytest.mark.parametrize(
    "layer_class_name, layer_kwargs, input_shape_wo_batchsize",
    [
        ("Dense", {"units": 3}, (1,)),
        ("Activation", {"activation": "relu"}, (1,)),
        ("Permute", {"dims": (1, 2, 3)}, (1, 1, 1)),
    ],
)
def test_preprocess_layer_no_nonlinear_activation(layer_class_name, layer_kwargs, input_shape_wo_batchsize):
    layer_class = globals()[layer_class_name]
    layer = layer_class(**layer_kwargs)
    # build layer
    input_tensor = Input(input_shape_wo_batchsize)
    layer(input_tensor)
    # preprocess
    layers = preprocess_layer(layer)
    # check resulting layers
    assert len(layers) == 1
    keras_layer = layers[0]
    # check values
    assert keras_layer is layer
    # try to call the resulting layers 3 times
    input_tensor = K.ones((5,) + input_shape_wo_batchsize)
    for _ in range(3):
        keras_layer(input_tensor)


@pytest.mark.parametrize(
    "layer_class_name, "
    "layer_kwargs, "
    "input_shape_wo_batchsize, "
    "embedded_activation_layer_class_name, "
    "embedded_activation_layer_class_kwargs, ",
    [
        ("Dense", {"units": 3, "activation": "relu"}, (1,), None, None),
        ("Dense", {"units": 3}, (1,), "PReLU", {}),
        ("Conv2D", {"filters": 2, "kernel_size": (3, 3), "activation": "relu"}, (64, 64, 3), None, None),
    ],
)
def test_preprocess_layer_nonlinear_activation(
    layer_class_name,
    layer_kwargs,
    input_shape_wo_batchsize,
    embedded_activation_layer_class_name,
    embedded_activation_layer_class_kwargs,
    use_bias,
):
    # init layer
    layer_class = globals()[layer_class_name]
    if embedded_activation_layer_class_name is not None:
        embedded_activation_layer_class = globals()[embedded_activation_layer_class_name]
        embedded_activation_layer = embedded_activation_layer_class(**embedded_activation_layer_class_kwargs)
        layer = layer_class(use_bias=use_bias, activation=embedded_activation_layer, **layer_kwargs)
    else:
        layer = layer_class(use_bias=use_bias, **layer_kwargs)
    # init input_shape and weights
    input_shape = input_shape_wo_batchsize
    input_tensor = Input(input_shape)
    layer(input_tensor)
    # split
    layers = preprocess_layer(layer)
    # check layer split
    assert len(layers) == 2
    layer_wo_activation, activation_layer = layers
    assert isinstance(layer_wo_activation, layer.__class__)
    assert layer_wo_activation.get_config()["activation"] == "linear"
    if isinstance(layer.activation, Layer):
        assert activation_layer == layer.activation
    else:
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
    output_np_ref = K.convert_to_numpy(layer(inputs_np))
    output_np_new = K.convert_to_numpy(activation_layer(layer_wo_activation(inputs_np)))
    assert_almost_equal(output_np_new, output_np_ref)
    # check same weights (really same objects)
    for i in range(len(layer_wo_activation.weights)):
        assert layer.weights[i] is layer_wo_activation.weights[i]
