# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Conv2D

from decomon.layers.decomon_layers import DecomonConv2D, to_monotonic


def test_Decomon_conv_box(data_format, mode, floatx, helpers):

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    monotonic_layer = DecomonConv2D(
        10, kernel_size=(3, 3), activation="relu", dc_decomp=True, mode=mode, data_format=data_format, dtype=K.floatx()
    )

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
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

    helpers.assert_output_properties_box(
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

    helpers.assert_output_properties_box(
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


def test_Decomon_conv_box_nodc(data_format, floatx, helpers):

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    monotonic_layer = DecomonConv2D(10, kernel_size=(3, 3), activation="relu", dc_decomp=False, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_layer(inputs[2:])

    W_, bias = monotonic_layer.get_weights()
    monotonic_layer.set_weights([np.maximum(0.0, W_), bias])
    f_conv = K.function(inputs[2:], output)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_conv(inputs_[2:])
    helpers.assert_output_properties_box_linear(
        x, None, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc", decimal=decimal
    )
    monotonic_layer.set_weights([np.minimum(0.0, W_), bias])
    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_conv(inputs_[2:])
    helpers.assert_output_properties_box_linear(
        x, None, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc", decimal=decimal
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


def test_Decomon_conv_to_monotonic_box(shared, floatx, helpers):
    data_format = "channels_last"
    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 4
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    conv_ref = Conv2D(10, kernel_size=(3, 3), activation="relu", dtype=K.floatx())
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
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

    helpers.assert_output_properties_box(
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


def test_Decomon_conv_to_monotonic_box_nodc(helpers):
    data_format = "channels_last"
    odd, m_0, m_1 = 0, 0, 1

    conv_ref = Conv2D(10, kernel_size=(3, 3), activation="relu")
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
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

    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )
