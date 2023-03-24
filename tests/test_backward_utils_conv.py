# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from numpy.testing import assert_almost_equal
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model

from decomon.backward_layers.backward_layers import get_backward
from decomon.backward_layers.utils_conv import get_toeplitz
from decomon.layers.decomon_layers import DecomonDense, to_monotonic


def test_toeplitz_from_Keras(channels, filter_size, strides, flatten, data_format, padding, floatx, helpers):

    # filter_size, strides, flatten,
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 0

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    # filter_size=3
    # strides=1
    # flatten = True
    odd, m_0, m_1 = 0, 0, 1

    # should be working either with convolution of conv2D
    layer = Conv2D(
        channels, (filter_size, filter_size), strides=strides, use_bias=False, padding=padding, dtype=K.floatx()
    )

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    y = inputs[1]
    result_ref = layer(y)
    W = get_toeplitz(layer, flatten)

    if not flatten:
        w_in, h_in, c_in, w_out, h_out, c_out = W.shape
        W = K.reshape(W, (w_in * h_in * c_in, w_out * h_out * c_out))

    n_in, n_out = W.shape
    y_flat = K.reshape(y, (-1, n_in, 1))
    result_flat = K.reshape(result_ref, (-1, n_out))
    result_toeplitz = K.sum(W[None] * y_flat, 1)
    output_test = K.sum((result_toeplitz - result_flat) ** 2)
    f_test = K.function(y, output_test)

    output_test_ = f_test(inputs_[1])
    assert_almost_equal(
        output_test_,
        np.zeros_like(output_test_),
        decimal=decimal,
        err_msg="wrong toeplitz matrix",
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


def test_toeplitz_from_Decomon(channels, filter_size, strides, flatten, data_format, padding, floatx, helpers):

    # filter_size, strides, flatten,
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 0

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    # filter_size=3
    # strides=1
    # flatten = True
    odd, m_0, m_1 = 0, 0, 1

    # should be working either with convolution of conv2D
    layer = Conv2D(
        channels, (filter_size, filter_size), strides=strides, use_bias=False, padding=padding, dtype=K.floatx()
    )

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    y = inputs[1]
    result_ref = layer(y)
    # toeplitz matrix should be compatible with a DecomonLayer
    input_dim = inputs[0].shape[-1]
    decomon_layer = to_monotonic(layer, input_dim)[0]

    W = get_toeplitz(decomon_layer, flatten)

    if not flatten:
        w_in, h_in, c_in, w_out, h_out, c_out = W.shape
        W = K.reshape(W, (w_in * h_in * c_in, w_out * h_out * c_out))

    n_in, n_out = W.shape
    y_flat = K.reshape(y, (-1, n_in, 1))
    result_flat = K.reshape(result_ref, (-1, n_out))
    result_toeplitz = K.sum(W[None] * y_flat, 1)
    output_test = K.sum((result_toeplitz - result_flat) ** 2)
    f_test = K.function(y, output_test)

    output_test_ = f_test(inputs_[1])
    assert_almost_equal(
        output_test_,
        np.zeros_like(output_test_),
        decimal=decimal,
        err_msg="wrong toeplitz matrix",
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
