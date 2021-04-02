# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from . import (
    get_standard_values_images_box,
    get_tensor_decomposition_images_box,
    assert_output_properties_box,
    assert_output_properties_box_linear,
)
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Conv2D
from decomon.layers.decomon_layers import DecomonConv2D, to_monotonic


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_Decomon_conv_box(data_format, odd, m_0, m_1):

    monotonic_layer = DecomonConv2D(10, kernel_size=(3, 3), activation="relu", dc_decomp=True)

    inputs = get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output = monotonic_layer(inputs[1:])

    W_, W_pos, W_neg, bias = monotonic_layer.get_weights()
    monotonic_layer.set_weights([W_pos, W_pos, np.zeros_like(W_neg), bias])
    f_conv = K.function(inputs[1:], output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_conv(inputs_[1:])

    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "conv_{}_{}_{}_{}".format(data_format, odd, m_0, m_1),
        decimal=5,
    )
    monotonic_layer.set_weights([W_neg, np.zeros_like(W_pos), W_neg, bias])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_conv(inputs_[1:])

    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "conv_{}_{}_{}_{}".format(data_format, odd, m_0, m_1),
        decimal=5,
    )


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_Decomon_conv_box_nodc(data_format, odd, m_0, m_1):

    monotonic_layer = DecomonConv2D(10, kernel_size=(3, 3), activation="relu", dc_decomp=False)

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_layer(inputs[1:])

    W_, W_pos, W_neg, bias = monotonic_layer.get_weights()
    monotonic_layer.set_weights([W_pos, W_pos, np.zeros_like(W_neg), bias])
    f_conv = K.function(inputs[1:], output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_conv(inputs_[1:])
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, 'nodc')
    monotonic_layer.set_weights([W_neg, np.zeros_like(W_pos), W_neg, bias])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_conv(inputs_[1:])
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, 'nodc')


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_Decomon_conv_to_monotonic_box(data_format, odd, m_0, m_1):

    conv_ref = Conv2D(10, kernel_size=(3, 3), activation="relu")
    inputs = get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output_ref = conv_ref(inputs[1])

    input_dim = x.shape[-1]
    monotonic_layer = to_monotonic(conv_ref, input_dim, dc_decomp=True)

    output = monotonic_layer(inputs[1:])
    f_ref = K.function(inputs, output_ref)

    f_conv = K.function(inputs[1:], output)
    y_ref = f_ref(inputs_)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_conv(inputs_[1:])

    assert np.allclose(y_, y_ref)
    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "conv_{}_{}_{}_{}".format(data_format, odd, m_0, m_1),
        decimal=4,
    )


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_Decomon_conv_to_monotonic_box_nodc(data_format, odd, m_0, m_1):

    conv_ref = Conv2D(10, kernel_size=(3, 3), activation="relu")
    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output_ref = conv_ref(inputs[1])

    input_dim = x.shape[-1]
    monotonic_layer = to_monotonic(conv_ref, input_dim, dc_decomp=False)

    output = monotonic_layer(inputs[1:])
    f_ref = K.function(inputs, output_ref)

    f_conv = K.function(inputs[1:], output)
    y_ref = f_ref(inputs_)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_conv(inputs_[1:])

    assert np.allclose(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, 'nodc')
