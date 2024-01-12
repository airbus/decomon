import keras.ops as K
import numpy as np
import pytest
from keras.layers import Activation, Conv2D, Dense, Flatten, Input, PReLU
from keras.models import Model, Sequential
from numpy.testing import assert_almost_equal

from decomon.models.convert import (
    preprocess_keras_model,
    split_activations_in_keras_model,
)


def test_split_activations_in_keras_model_no_inputshape_ko():
    layers = [
        Conv2D(
            10,
            kernel_size=(3, 3),
            activation="relu",
            data_format="channels_last",
        ),
        Flatten(),
        Dense(1),
    ]
    model = Sequential(layers)
    with pytest.raises(ValueError):
        converted_model = split_activations_in_keras_model(model)


def test_split_activations_in_keras_model(toy_model):
    converted_model = split_activations_in_keras_model(toy_model)
    assert isinstance(converted_model, Model)
    # check no more activation functions in non-activation layers
    for layer in converted_model.layers:
        activation = layer.get_config().get("activation", None)
        assert isinstance(layer, Activation) or activation is None or activation == "linear"
    # check same outputs
    input_shape_wo_batchsize = toy_model.input_shape[1:]
    input_shape_with_batch_size = (5,) + input_shape_wo_batchsize
    flatten_dim = np.prod(input_shape_with_batch_size)
    inputs_np = np.linspace(-1, 1, flatten_dim).reshape(input_shape_with_batch_size)
    output_np_ref = K.convert_to_numpy(toy_model(inputs_np))
    output_np_new = K.convert_to_numpy(converted_model(inputs_np))
    assert_almost_equal(output_np_new, output_np_ref, decimal=4)


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
def test_preprocess(
    layer_class_name,
    layer_kwargs,
    input_shape_wo_batchsize,
    embedded_activation_layer_class_name,
    embedded_activation_layer_class_kwargs,
    use_bias,
):
    # init hidden layer
    layer_class = globals()[layer_class_name]
    if embedded_activation_layer_class_name is not None:
        embedded_activation_layer_class = globals()[embedded_activation_layer_class_name]
        embedded_activation_layer = embedded_activation_layer_class(**embedded_activation_layer_class_kwargs)
        hidden_layer = layer_class(use_bias=use_bias, activation=embedded_activation_layer, **layer_kwargs)
    else:
        hidden_layer = layer_class(use_bias=use_bias, **layer_kwargs)
    layers = [
        Input(shape=input_shape_wo_batchsize),
        hidden_layer,
        Dense(1),
    ]
    model = Sequential(layers)
    converted_model = preprocess_keras_model(model)
    # check no more embedded activation
    for layer in converted_model.layers:
        activation = layer.get_config().get("activation", None)
        assert isinstance(layer, Activation) or activation is None or activation == "linear"
    # check number of layers
    assert len(converted_model.layers) == 4
    # check same outputs
    input_shape_wo_batchsize = model.input_shape[1:]
    input_shape_with_batch_size = (5,) + input_shape_wo_batchsize
    flatten_dim = np.prod(input_shape_with_batch_size)
    inputs_np = np.linspace(-1, 1, flatten_dim).reshape(input_shape_with_batch_size)
    output_np_ref = K.convert_to_numpy(model(inputs_np))
    output_np_new = K.convert_to_numpy(converted_model(inputs_np))
    assert_almost_equal(output_np_new, output_np_ref, decimal=4)
