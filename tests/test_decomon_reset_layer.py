import keras_core as keras
import numpy as np
import pytest
from keras_core.layers import BatchNormalization, Conv2D, Dense, Flatten
from numpy.testing import assert_almost_equal

from decomon.core import ForwardMode
from decomon.layers.decomon_layers import (
    DecomonBatchNormalization,
    DecomonConv2D,
    DecomonDense,
    DecomonFlatten,
)


def test_decomondense_reset_layer(helpers, use_bias):
    dc_decomp = False
    input_dim = 1
    units = 3
    mode = ForwardMode.HYBRID
    layer = Dense(units=units, use_bias=use_bias)
    layer(K.zeros((2, input_dim)))
    decomon_layer = DecomonDense(units=units, use_bias=use_bias, mode=mode, dc_decomp=dc_decomp)
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    decomon_layer(inputs_for_mode)

    kernel = layer.kernel
    layer.kernel.assign(2 * np.ones_like(kernel))
    decomon_layer.kernel.assign(np.zeros_like(kernel))
    if use_bias:
        bias = layer.bias
        layer.bias.assign(np.ones_like(bias))
        decomon_layer.bias.assign(np.zeros_like(bias))

    decomon_layer.reset_layer(layer)
    assert decomon_layer.kernel is not layer.kernel
    assert_almost_equal(decomon_layer.kernel.numpy(), layer.kernel.numpy())
    if use_bias:
        assert len(layer.weights) == 2
        assert decomon_layer.bias is not layer.bias
        assert_almost_equal(decomon_layer.bias.numpy(), layer.bias.numpy())
    else:
        assert len(layer.weights) == 1


def test_decomondense_reset_layer_decomon_with_new_weights(helpers):
    dc_decomp = False
    input_dim = 1
    units = 3
    mode = ForwardMode.HYBRID
    layer = Dense(units=units)
    layer(K.zeros((2, input_dim)))
    decomon_layer = DecomonDense(units=units, mode=mode, dc_decomp=dc_decomp)
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    decomon_layer(inputs_for_mode)
    # Add new variables
    decomon_layer.add_weight(shape=(input_dim, units), initializer="ones", name="alpha", trainable=True)
    decomon_layer.add_weight(shape=(input_dim, units), initializer="ones", name="beta", trainable=False)
    assert len(decomon_layer.weights) == 4
    assert len(decomon_layer.trainable_weights) == 3

    kernel, bias = layer.get_weights()
    layer.set_weights([2 * np.ones_like(kernel), np.ones_like(bias)])
    decomon_layer.kernel.assign(np.zeros_like(kernel))
    decomon_layer.bias.assign(np.zeros_like(bias))

    decomon_layer.reset_layer(layer)
    assert decomon_layer.kernel is not layer.kernel
    assert decomon_layer.bias is not layer.bias
    assert_almost_equal(decomon_layer.kernel.numpy(), layer.kernel.numpy())
    assert_almost_equal(decomon_layer.bias.numpy(), layer.bias.numpy())


def test_decomondense_reset_layer_keras_with_new_weights(helpers):
    dc_decomp = False
    input_dim = 1
    units = 3
    mode = ForwardMode.HYBRID
    layer = Dense(units=units)
    layer(K.zeros((2, input_dim)))
    decomon_layer = DecomonDense(units=units, mode=mode, dc_decomp=dc_decomp)
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    decomon_layer(inputs_for_mode)
    # Add new variables
    layer.add_weight(shape=(input_dim, units), initializer="ones", name="alpha", trainable=False)
    assert len(layer.weights) == 3

    kernel = layer.kernel
    layer.kernel.assign(2 * np.ones_like(kernel))
    decomon_layer.kernel.assign(np.zeros_like(kernel))
    bias = layer.bias
    layer.bias.assign(np.ones_like(bias))
    decomon_layer.bias.assign(np.zeros_like(bias))

    decomon_layer.reset_layer(layer)
    assert decomon_layer.kernel is not layer.kernel
    assert decomon_layer.bias is not layer.bias
    assert_almost_equal(decomon_layer.kernel.numpy(), layer.kernel.numpy())
    assert_almost_equal(decomon_layer.bias.numpy(), layer.bias.numpy())


def test_decomondense_reset_layer_ko_keraslayer_not_nuilt():
    dc_decomp = False
    input_dim = 1
    units = 3
    mode = ForwardMode.HYBRID
    layer = Dense(units=units)
    decomon_layer = DecomonDense(units=units, mode=mode, dc_decomp=dc_decomp)
    with pytest.raises(ValueError):
        decomon_layer.reset_layer(layer)


def test_decomondense_reset_layer_ko_decomonlayer_not_nuilt():
    dc_decomp = False
    input_dim = 1
    units = 3
    mode = ForwardMode.HYBRID
    layer = Dense(units=units)
    layer(K.zeros((2, input_dim)))
    decomon_layer = DecomonDense(units=units, mode=mode, dc_decomp=dc_decomp)
    with pytest.raises(ValueError):
        decomon_layer.reset_layer(layer)


def test_decomonconv2d_reset_layer(helpers, use_bias):
    dc_decomp = False
    odd = 0
    data_format = "channels_last"
    filters = 3
    kernel_size = (3, 3)
    mode = ForwardMode.HYBRID

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    layer = Conv2D(filters=filters, kernel_size=kernel_size, use_bias=use_bias)
    layer(input_ref)
    decomon_layer = DecomonConv2D(
        filters=filters, kernel_size=kernel_size, use_bias=use_bias, mode=mode, dc_decomp=dc_decomp
    )
    decomon_layer(inputs_for_mode)

    kernel = layer.kernel
    layer.kernel.assign(2 * np.ones_like(kernel))
    decomon_layer.kernel.assign(np.zeros_like(kernel))
    if use_bias:
        bias = layer.bias
        layer.bias.assign(np.ones_like(bias))
        decomon_layer.bias.assign(np.zeros_like(bias))

    decomon_layer.reset_layer(layer)
    assert decomon_layer.kernel is not layer.kernel
    assert_almost_equal(decomon_layer.kernel.numpy(), layer.kernel.numpy())
    if use_bias:
        assert decomon_layer.bias is not layer.bias
        assert_almost_equal(decomon_layer.bias.numpy(), layer.bias.numpy())


@pytest.mark.parametrize(
    "center, scale",
    [
        (True, True),
        (False, False),
    ],
)
def test_decomonbacthnormalization_reset_layer(helpers, center, scale):
    dc_decomp = False
    odd = 0
    mode = ForwardMode.HYBRID
    inputs = helpers.get_tensor_decomposition_multid_box(odd=odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    layer = BatchNormalization(center=center, scale=scale)
    layer(input_ref)
    decomon_layer = DecomonBatchNormalization(center=center, scale=scale, mode=mode, dc_decomp=dc_decomp)
    decomon_layer(inputs_for_mode)

    moving_mean = layer.moving_mean
    layer.moving_mean.assign(np.ones_like(moving_mean))
    decomon_layer.moving_mean.assign(np.zeros_like(moving_mean))
    moving_variance = layer.moving_variance
    layer.moving_variance.assign(np.ones_like(moving_variance))
    decomon_layer.moving_variance.assign(np.zeros_like(moving_variance))
    if center:
        beta = layer.beta
        layer.beta.assign(np.ones_like(beta))
        decomon_layer.beta.assign(np.zeros_like(beta))
    if scale:
        gamma = layer.gamma
        layer.gamma.assign(np.ones_like(gamma))
        decomon_layer.gamma.assign(np.zeros_like(gamma))

    decomon_layer.reset_layer(layer)

    keras_weights = layer.get_weights()
    decomon_weights = layer.get_weights()
    for i in range(len(decomon_weights)):
        assert_almost_equal(decomon_weights[i], keras_weights[i])


def test_decomonflatten_reset_layer(helpers):
    dc_decomp = False
    input_dim = 1
    mode = ForwardMode.HYBRID
    layer = Flatten()
    layer(K.zeros((2, input_dim)))
    decomon_layer = DecomonFlatten(mode=mode, dc_decomp=dc_decomp)
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    decomon_layer(inputs_for_mode)

    decomon_layer.reset_layer(layer)
