import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal
from tensorflow.keras import Input
from tensorflow.keras.layers import Activation, Conv2D, Dense, Permute, PReLU

from decomon.models.utils import convert_deellip_to_keras, split_activation

try:
    import deel.lip
except ImportError:
    deel_lip_available = False
else:
    deel_lip_available = True
    from deel.lip.activations import GroupSort
    from deel.lip.layers import SpectralDense


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


def test_split_activation_embedded_layer():
    # init layer
    prelu_layer = PReLU()
    layer = Dense(units=3, activation=prelu_layer)
    # build layer
    input_shape_wo_batchsize = (1,)
    input_tensor = Input(input_shape_wo_batchsize)
    layer(input_tensor)
    # split
    layers = split_activation(layer)
    # check layer split
    assert len(layers) == 2
    layer_wo_activation, activation_layer = layers
    assert isinstance(layer_wo_activation, layer.__class__)
    assert layer_wo_activation.get_config()["activation"] == "linear"
    assert activation_layer is prelu_layer
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


@pytest.mark.skipif(not (deel_lip_available), reason="deel.lip is not available")
def test_split_activation_embedded_layer_with_deellip():
    # init layer
    groupsort_layer = GroupSort(n=1)
    layer = SpectralDense(units=3, activation=groupsort_layer)
    # build layer
    input_shape_wo_batchsize = (1,)
    input_tensor = Input(input_shape_wo_batchsize)
    layer(input_tensor)
    # split
    layers = split_activation(layer)
    # check layer split
    assert len(layers) == 2
    layer_wo_activation, activation_layer = layers
    assert isinstance(layer_wo_activation, layer.__class__)
    assert layer_wo_activation.get_config()["activation"] == "linear"
    assert activation_layer is groupsort_layer
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


def test_convert_deellip_to_keras_dense():
    layer = Dense(units=3)
    keras_layer = convert_deellip_to_keras(layer)
    # new attributes added
    assert hasattr(keras_layer, "is_lipschitz")
    assert hasattr(keras_layer, "deellip_classname")
    assert hasattr(keras_layer, "k_coef_lip")
    # check values
    assert not keras_layer.is_lipschitz
    assert keras_layer.deellip_classname == layer.__class__.__name__
    assert keras_layer.k_coef_lip == -1.0
    assert keras_layer is layer
    assert keras_layer.name.startswith(layer.name)
    # idempotency
    keras_layer2 = convert_deellip_to_keras(keras_layer)
    assert keras_layer is keras_layer2
    assert keras_layer.is_lipschitz == keras_layer2.is_lipschitz
    assert keras_layer.deellip_classname == keras_layer2.deellip_classname
    assert keras_layer.k_coef_lip == keras_layer2.k_coef_lip


@pytest.mark.skipif(not (deel_lip_available), reason="deel.lip is not available")
def test_convert_deellip_to_keras_groupsort():
    layer = GroupSort(n=2)
    keras_layer = convert_deellip_to_keras(layer)
    # new attributes added
    assert hasattr(keras_layer, "is_lipschitz")
    assert hasattr(keras_layer, "deellip_classname")
    assert hasattr(keras_layer, "k_coef_lip")
    # check values
    assert keras_layer.is_lipschitz
    assert keras_layer.deellip_classname == layer.__class__.__name__
    assert keras_layer.k_coef_lip == layer.k_coef_lip
    assert keras_layer is layer
    assert keras_layer.name.startswith(layer.name)
    # idempotency
    keras_layer2 = convert_deellip_to_keras(keras_layer)
    assert keras_layer is keras_layer2
    assert keras_layer.is_lipschitz == keras_layer2.is_lipschitz
    assert keras_layer.deellip_classname == keras_layer2.deellip_classname
    assert keras_layer.k_coef_lip == keras_layer2.k_coef_lip


@pytest.mark.skipif(not (deel_lip_available), reason="deel.lip is not available")
def test_convert_deellip_to_keras_spectraldense():
    units = 3
    layer = SpectralDense(units=units)
    layer(
        Input(
            1,
        )
    )
    keras_layer = convert_deellip_to_keras(layer)
    # new attributes added
    assert hasattr(keras_layer, "is_lipschitz")
    assert hasattr(keras_layer, "deellip_classname")
    assert hasattr(keras_layer, "k_coef_lip")
    # check values
    assert keras_layer.is_lipschitz
    assert keras_layer.deellip_classname == layer.__class__.__name__
    assert keras_layer.k_coef_lip == layer.k_coef_lip
    assert isinstance(keras_layer, Dense)
    assert keras_layer.units == units
    assert keras_layer.name.startswith(layer.name)
    # same output?
    input_tensor = tf.ones((4, 1))
    output_ref = layer(input_tensor).numpy()
    new_output = keras_layer(input_tensor).numpy()
    assert_almost_equal(new_output, output_ref)
    # idempotency
    keras_layer2 = convert_deellip_to_keras(keras_layer)
    assert keras_layer is keras_layer2
    assert keras_layer.is_lipschitz == keras_layer2.is_lipschitz
    assert keras_layer.deellip_classname == keras_layer2.deellip_classname
    assert keras_layer.k_coef_lip == keras_layer2.k_coef_lip


@pytest.mark.skipif(not (deel_lip_available), reason="deel.lip is not available")
def test_convert_deellip_to_keras_spectraldense_not_initialized_ko():
    units = 3
    layer = SpectralDense(units=units)
    with pytest.raises(RuntimeError):
        keras_layer = convert_deellip_to_keras(layer)
