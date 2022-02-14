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


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, mode, floatx",
    [
        ("channels_last", 0, 0, 1, "hybrid", 32),
        ("channels_last", 0, 0, 1, "forward", 32),
        ("channels_last", 0, 0, 1, "ibp", 32),
        ("channels_last", 0, 0, 1, "hybrid", 64),
        ("channels_last", 0, 0, 1, "forward", 64),
        ("channels_last", 0, 0, 1, "ibp", 64),
        ("channels_last", 0, 0, 1, "hybrid", 16),
        ("channels_last", 0, 0, 1, "forward", 16),
        ("channels_last", 0, 0, 1, "ibp", 16),
        ("channels_first", 0, 0, 1, "hybrid", 32),
        ("channels_first", 0, 0, 1, "forward", 32),
        ("channels_first", 0, 0, 1, "ibp", 32),
        ("channels_first", 0, 0, 1, "hybrid", 64),
        ("channels_first", 0, 0, 1, "forward", 64),
        ("channels_first", 0, 0, 1, "ibp", 64),
        ("channels_first", 0, 0, 1, "hybrid", 16),
        ("channels_first", 0, 0, 1, "forward", 16),
        ("channels_first", 0, 0, 1, "ibp", 16),
    ],
)
def test_Decomon_conv_box(data_format, odd, m_0, m_1, mode, floatx):

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    monotonic_layer = DecomonConv2D(
        10, kernel_size=(3, 3), activation="relu", dc_decomp=True, mode=mode, data_format=data_format
    )

    inputs = get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    if mode == "hybrid":
        output = monotonic_layer(inputs[2:])
    if mode == "forward":
        output = monotonic_layer([z, W_u, b_u, W_l, b_l, h, g])
    if mode == "ibp":
        output = monotonic_layer([u_c, l_c, h, g])

    W_, bias = monotonic_layer.get_weights()
    monotonic_layer.set_weights([np.maximum(0.0, W_), bias])
    f_conv = K.function(inputs[2:], output)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_conv(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_conv(inputs_[2:])
        u_c_ = None
        l_c_ = None
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_conv(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_,
        None,
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
        decimal=decimal,
    )
    monotonic_layer.set_weights([np.minimum(0.0, W_), bias])

    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_conv(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_conv(inputs_[2:])
        u_c_ = None
        l_c_ = None
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_conv(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_,
        None,
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
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, floatx",
    [
        ("channels_last", 0, 0, 1, 32),
        ("channels_last", 0, 0, 1, 64),
        ("channels_last", 0, 0, 1, 16),
        ("channels_first", 0, 0, 1, 32),
        ("channels_first", 0, 0, 1, 64),
        ("channels_first", 0, 0, 1, 16),
    ],
)
def test_Decomon_conv_box_nodc(data_format, odd, m_0, m_1, floatx):

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    monotonic_layer = DecomonConv2D(10, kernel_size=(3, 3), activation="relu", dc_decomp=False)

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_layer(inputs[2:])

    W_, bias = monotonic_layer.get_weights()
    monotonic_layer.set_weights([np.maximum(0.0, W_), bias])
    f_conv = K.function(inputs[2:], output)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_conv(inputs_[2:])
    assert_output_properties_box_linear(
        x, None, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc", decimal=decimal
    )
    monotonic_layer.set_weights([np.minimum(0.0, W_), bias])
    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_conv(inputs_[2:])
    assert_output_properties_box_linear(
        x, None, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc", decimal=decimal
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, shared, floatx",
    [
        ("channels_last", 0, 0, 1, False, 32),
        ("channels_last", 0, 0, 1, True, 32),
        ("channels_last", 0, 0, 1, False, 64),
        ("channels_last", 0, 0, 1, True, 64),
        ("channels_last", 0, 0, 1, False, 16),
        ("channels_last", 0, 0, 1, True, 16),
    ],
)
def test_Decomon_conv_to_monotonic_box(data_format, odd, m_0, m_1, shared, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 4
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    conv_ref = Conv2D(10, kernel_size=(3, 3), activation="relu")
    inputs = get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output_ref = conv_ref(inputs[1])

    input_dim = x.shape[-1]
    monotonic_layer = to_monotonic(conv_ref, input_dim, dc_decomp=True, shared=shared)

    output = monotonic_layer[0](inputs[2:])
    if len(monotonic_layer) > 1:
        output = monotonic_layer[1](output)

    f_ref = K.function(inputs, output_ref)

    f_conv = K.function(inputs[2:], output)
    y_ref = f_ref(inputs_)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_conv(inputs_[2:])

    assert_output_properties_box(
        x,
        y_ref,
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
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_Decomon_conv_to_monotonic_box_nodc(data_format, odd, m_0, m_1):

    conv_ref = Conv2D(10, kernel_size=(3, 3), activation="relu")
    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output_ref = conv_ref(inputs[1])

    input_dim = x.shape[-1]
    monotonic_layer = to_monotonic(conv_ref, input_dim, dc_decomp=False)

    output = monotonic_layer[0](inputs[2:])
    if len(monotonic_layer) > 1:
        output = monotonic_layer[1](output)

    f_ref = K.function(inputs, output_ref)

    f_conv = K.function(inputs[2:], output)
    y_ref = f_ref(inputs_)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_conv(inputs_[2:])

    assert_output_properties_box_linear(x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")
