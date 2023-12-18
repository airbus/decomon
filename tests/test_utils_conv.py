# Test unit for decomon with Dense layers
import keras.config
import keras.ops as K
import numpy as np
import pytest
from keras.layers import Conv2D
from numpy.testing import assert_almost_equal

from decomon.backward_layers.utils_conv import get_toeplitz
from decomon.core import get_affine, get_ibp
from decomon.layers.convert import to_decomon


def test_toeplitz_from_Keras(channels, filter_size, strides, flatten, data_format, padding, floatx, decimal, helpers):
    # skip unavailable combinations
    if floatx == 16 and keras.config.backend() == "torch":
        pytest.skip("Pytorch does not implement conv2d for float16")

    # filter_size, strides, flatten,
    if floatx == 16:
        decimal = 0

    if data_format == "channels_first" and keras.config.backend() == "tensorflow" and not helpers.in_GPU_mode():
        pytest.skip("data format 'channels first' is possible only in GPU mode for tensorflow")

    if data_format == "channels_first":
        pytest.xfail("get_toeplitz with channels_first is bugged for now.")

    dc_decomp = False
    odd, m_0, m_1 = 0, 0, 1

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)

    # should be working either with convolution of conv2D
    kwargs_layer = dict(
        data_format=data_format,
        filters=channels,
        kernel_size=(filter_size, filter_size),
        strides=strides,
        padding=padding,
        use_bias=False,
        dtype=input_ref.dtype,
    )

    layer = Conv2D(**kwargs_layer)

    result_ref = layer(input_ref)
    W = get_toeplitz(layer, flatten)

    if not flatten:
        w_in, h_in, c_in, w_out, h_out, c_out = W.shape
        W = K.reshape(W, (w_in * h_in * c_in, w_out * h_out * c_out))

    n_in, n_out = W.shape
    y_flat = K.reshape(input_ref, (-1, n_in, 1))
    result_flat = K.reshape(result_ref, (-1, n_out))
    result_toeplitz = K.sum(W[None] * y_flat, 1)
    output_test = K.sum((result_toeplitz - result_flat) ** 2)
    f_test = helpers.function(inputs, output_test)
    output_test_ = f_test(inputs_)

    assert_almost_equal(
        output_test_,
        np.zeros_like(output_test_),
        decimal=decimal,
        err_msg="wrong toeplitz matrix",
    )


def test_toeplitz_from_Decomon(
    floatx, decimal, mode, channels, filter_size, strides, flatten, data_format, padding, helpers
):
    # skip unavailable combinations
    if floatx == 16 and keras.config.backend() == "torch" and not helpers.in_GPU_mode():
        pytest.skip("Pytorch does not implement conv2d for float16 in CPU mode.")

    if data_format == "channels_first" and keras.config.backend() == "tensorflow" and not helpers.in_GPU_mode():
        pytest.skip("data format 'channels first' is possible only in GPU mode for tensorflow")

    if data_format == "channels_first":
        pytest.xfail("get_toeplitz with channels_first is bugged for now.")

    odd, m_0, m_1 = 0, 0, 1
    if floatx == 16:
        decimal = 0

    dc_decomp = False
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)
    kwargs_layer = dict(
        filters=channels,
        kernel_size=(filter_size, filter_size),
        strides=strides,
        padding=padding,
        use_bias=False,
        dtype=input_ref.dtype,
        data_format=data_format,
    )

    # should be working either with convolution of conv2D
    layer = Conv2D(**kwargs_layer)
    result_ref = layer(input_ref)

    # toeplitz matrix should be compatible with a DecomonLayer
    input_dim = input_ref.shape[-1]

    decomon_layer = to_decomon(layer, input_dim, ibp=ibp, affine=affine)

    W = get_toeplitz(decomon_layer, flatten)

    if not flatten:
        w_in, h_in, c_in, w_out, h_out, c_out = W.shape
        W = K.reshape(W, (w_in * h_in * c_in, w_out * h_out * c_out))

    n_in, n_out = W.shape
    y_flat = K.reshape(input_ref, (-1, n_in, 1))
    result_flat = K.reshape(result_ref, (-1, n_out))
    result_toeplitz = K.sum(W[None] * y_flat, 1)

    output_test = K.sum((result_toeplitz - result_flat) ** 2)

    f_test = helpers.function(inputs, output_test)
    output_test_ = f_test(inputs_)

    assert_almost_equal(
        output_test_,
        np.zeros_like(output_test_),
        decimal=decimal,
        err_msg="wrong toeplitz matrix",
    )
