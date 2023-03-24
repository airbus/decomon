# Test unit for decomon with Dense layers


import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import MaxPooling2D

from decomon.layers.core import ForwardMode
from decomon.layers.maxpooling import DecomonMaxPooling2D


def test_MaxPooling2D_box(mode, floatx, helpers):
    odd, m_0, m_1 = 0, 0, 1
    data_format = "channels_last"
    fast = False

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_
    output_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", dtype=K.floatx())(inputs[1])
    f_ref = K.function(inputs, output_ref)

    layer = DecomonMaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", dc_decomp=True, fast=fast, mode=mode, dtype=K.floatx()
    )

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = layer(inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = layer([inputs[2], inputs[4], inputs[5], inputs[7], inputs[8], inputs[9], inputs[10]])
    elif mode == ForwardMode.IBP:
        output = layer([inputs[3], inputs[6], inputs[9], inputs[10]])
    else:
        raise ValueError("Unknown mode.")

    f_pooling = K.function(inputs, output)

    y_ = f_ref(inputs_)
    z_ = z
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_pooling(inputs_)
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_pooling(inputs_)
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_pooling(inputs_)
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
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
        "maxpooling_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
