import keras_core as keras
import keras_core.ops as K
import numpy as np
import pytest
from keras_core import Input
from keras_core.layers import Activation, Conv2D, Dense, Layer, Permute, PReLU
from numpy.testing import assert_almost_equal

from decomon.models.utils import (
    convert_deellip_to_keras,
    preprocess_layer,
    split_activation,
)

try:
    import deel.lip
except ImportError:
    deel_lip_available = False
else:
    deel_lip_available = True
    from deel.lip.activations import GroupSort
    from deel.lip.layers import SpectralDense

deel_lip_skip_reason = "deel-lip is not available"


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


@pytest.mark.skipif(not (deel_lip_available), reason=deel_lip_skip_reason)
@pytest.mark.parametrize(
    "layer_class_name, "
    "layer_kwargs, "
    "input_shape_wo_batchsize, "
    "embedded_activation_layer_class_name, "
    "embedded_activation_layer_kwargs",
    [
        ("SpectralDense", {"units": 3, "activation": "relu"}, (1,), None, None),
        ("Dense", {"units": 3}, (1,), "GroupSort", {"n": 1}),
    ],
)
def test_split_activation_do_split_with_deellip(
    layer_class_name,
    layer_kwargs,
    input_shape_wo_batchsize,
    embedded_activation_layer_class_name,
    embedded_activation_layer_kwargs,
    use_bias,
):
    # init layer
    layer_class = globals()[layer_class_name]
    if embedded_activation_layer_class_name is not None:
        embedded_activation_layer_class = globals()[embedded_activation_layer_class_name]
        layer = layer_class(
            use_bias=use_bias,
            activation=embedded_activation_layer_class(**embedded_activation_layer_kwargs),
            **layer_kwargs,
        )
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


@pytest.mark.skipif(not (deel_lip_available), reason=deel_lip_skip_reason)
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


@pytest.mark.skipif(not (deel_lip_available), reason=deel_lip_skip_reason)
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
    input_tensor = K.ones((4, 1))
    output_ref = layer(input_tensor).numpy()
    new_output = keras_layer(input_tensor).numpy()
    assert_almost_equal(new_output, output_ref)
    # idempotency
    keras_layer2 = convert_deellip_to_keras(keras_layer)
    assert keras_layer is keras_layer2
    assert keras_layer.is_lipschitz == keras_layer2.is_lipschitz
    assert keras_layer.deellip_classname == keras_layer2.deellip_classname
    assert keras_layer.k_coef_lip == keras_layer2.k_coef_lip


@pytest.mark.skipif(not (deel_lip_available), reason=deel_lip_skip_reason)
def test_convert_deellip_to_keras_spectraldense_not_initialized_ko():
    units = 3
    layer = SpectralDense(units=units)
    with pytest.raises(RuntimeError):
        keras_layer = convert_deellip_to_keras(layer)


@pytest.mark.parametrize(
    "layer_class_name, layer_kwargs, input_shape_wo_batchsize, is_deel_lip",
    [
        ("Dense", {"units": 3}, (1,), False),
        ("Activation", {"activation": "relu"}, (1,), False),
        ("Permute", {"dims": (1, 2, 3)}, (1, 1, 1), False),
        ("SpectralDense", {"units": 3}, (1,), True),
        ("GroupSort", {"n": 1}, (1,), True),
    ],
)
def test_preprocess_layer_no_nonlinear_activation(
    layer_class_name, layer_kwargs, input_shape_wo_batchsize, is_deel_lip
):
    if is_deel_lip and not deel_lip_available:
        pytest.skip(deel_lip_skip_reason)
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
    # new attributes added
    assert hasattr(keras_layer, "is_lipschitz")
    assert hasattr(keras_layer, "deellip_classname")
    assert hasattr(keras_layer, "k_coef_lip")
    # check values
    assert keras_layer.deellip_classname == layer.__class__.__name__
    if is_deel_lip:
        assert keras_layer.is_lipschitz
        assert keras_layer.k_coef_lip == layer.k_coef_lip
        if hasattr(layer, "vanilla_export"):
            assert keras_layer is not layer
        else:
            assert keras_layer is layer
    else:
        assert not keras_layer.is_lipschitz
        assert keras_layer.k_coef_lip == -1.0
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
    "embedded_activation_layer_class_kwargs, "
    "is_deellip_layer, "
    "is_deellip_activation",
    [
        ("Dense", {"units": 3, "activation": "relu"}, (1,), None, None, False, False),
        ("Dense", {"units": 3}, (1,), "PReLU", {}, False, False),
        ("Conv2D", {"filters": 2, "kernel_size": (3, 3), "activation": "relu"}, (64, 64, 3), None, None, False, False),
        ("Dense", {"units": 3}, (1,), "GroupSort", {"n": 1}, False, True),
        ("SpectralDense", {"units": 3, "activation": "relu"}, (1,), None, None, True, False),
        ("SpectralDense", {"units": 3}, (1,), "PReLU", {}, True, False),
        ("SpectralDense", {"units": 3}, (1,), "GroupSort", {"n": 1}, True, True),
    ],
)
def test_preprocess_layer_nonlinear_activation(
    layer_class_name,
    layer_kwargs,
    input_shape_wo_batchsize,
    embedded_activation_layer_class_name,
    embedded_activation_layer_class_kwargs,
    use_bias,
    is_deellip_layer,
    is_deellip_activation,
):
    # skip deel-lip cases if not available
    if not deel_lip_available and (is_deellip_layer or is_deellip_activation):
        pytest.skip(reason=deel_lip_skip_reason)
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
    if is_deellip_layer:
        if layer.__class__.__name__.endswith("Dense"):
            assert isinstance(layer_wo_activation, Dense)
        if layer.__class__.__name__.endswith("Conv2D"):
            assert isinstance(layer_wo_activation, Conv2D)
    else:
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
    output_np_ref = layer(inputs_np).numpy()
    output_np_new = activation_layer(layer_wo_activation(inputs_np)).numpy()
    assert_almost_equal(output_np_new, output_np_ref)
    # check same trainable weights
    if not is_deellip_layer:
        for i in range(len(layer._trainable_weights)):
            assert layer._trainable_weights[i] is layer_wo_activation._trainable_weights[i]
    # check deel-lip attributes
    for i, keras_layer in enumerate(layers):
        # new attributes added
        assert hasattr(keras_layer, "is_lipschitz")
        assert hasattr(keras_layer, "deellip_classname")
        assert hasattr(keras_layer, "k_coef_lip")
        # check values
        if i == 0:
            assert keras_layer.deellip_classname == layer.__class__.__name__
            if is_deellip_layer:
                assert keras_layer.is_lipschitz
                assert keras_layer.k_coef_lip == layer.k_coef_lip
            else:
                assert not keras_layer.is_lipschitz
                assert keras_layer.k_coef_lip == -1.0
        else:
            if isinstance(layer.activation, Layer):
                assert keras_layer.deellip_classname == layer.activation.__class__.__name__
            else:
                assert keras_layer.deellip_classname == Activation.__name__
            if is_deellip_activation:
                assert keras_layer.is_lipschitz
                assert keras_layer.k_coef_lip == layer.activation.k_coef_lip
            else:
                assert not keras_layer.is_lipschitz
                assert keras_layer.k_coef_lip == -1.0
