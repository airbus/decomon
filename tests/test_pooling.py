# Test unit for decomon with Dense layers


import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import MaxPooling2D

from decomon.layers.maxpooling import DecomonMaxPooling2D

from . import (
    assert_output_properties_box,
    get_standard_values_images_box,
    get_tensor_decomposition_images_box,
)


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, fast, mode, floatx",
    [
        ("channels_last", 0, 0, 1, False, "hybrid", 32),
        ("channels_last", 0, 0, 1, False, "forward", 32),
        ("channels_last", 0, 0, 1, False, "ibp", 32),
        ("channels_last", 0, 0, 1, False, "hybrid", 16),
        ("channels_last", 0, 0, 1, False, "forward", 16),
        ("channels_last", 0, 0, 1, False, "ibp", 16),
        ("channels_last", 0, 0, 1, False, "hybrid", 64),
        ("channels_last", 0, 0, 1, False, "forward", 64),
        ("channels_last", 0, 0, 1, False, "ibp", 64),
    ],
)
def test_MaxPooling2D_box(data_format, odd, m_0, m_1, fast, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_
    output_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", dtype=K.floatx())(inputs[1])
    f_ref = K.function(inputs, output_ref)

    layer = DecomonMaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", dc_decomp=True, fast=fast, mode=mode, dtype=K.floatx()
    )

    if mode == "hybrid":
        output = layer(inputs[2:])
    if mode == "forward":
        output = layer([inputs[2], inputs[4], inputs[5], inputs[7], inputs[8], inputs[9], inputs[10]])
    if mode == "ibp":
        output = layer([inputs[3], inputs[6], inputs[9], inputs[10]])

    f_pooling = K.function(inputs, output)

    y_ = f_ref(inputs_)
    z_ = z
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_pooling(inputs_)
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_pooling(inputs_)
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_pooling(inputs_)

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
        "maxpooling_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


"""
@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, fast, mode, floatx",
    [
        ("channels_last", 0, 0, 1, False, "hybrid", 32),
        ("channels_last", 0, 0, 1, False, "forward", 32),
        ("channels_last", 0, 0, 1, False, "ibp", 32),
        ("channels_last", 0, 0, 1, False, "hybrid", 16),
        ("channels_last", 0, 0, 1, False, "forward", 16),
        ("channels_last", 0, 0, 1, False, "ibp", 16),
        ("channels_last", 0, 0, 1, False, "hybrid", 64),
        ("channels_last", 0, 0, 1, False, "forward", 64),
        ("channels_last", 0, 0, 1, False, "ibp", 64),
    ],
)
def test_MaxPooling2D_box_nodc(data_format, odd, m_0, m_1, fast, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    output_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", dtype=K.floatx())(inputs[1])
    f_ref = K.function(inputs, output_ref)

    layer = DecomonMaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", dc_decomp=False, fast=fast, mode=mode, dtype=K.floatx()
    )

    if mode == "hybrid":
        output = layer(inputs[2:])
    if mode == "forward":
        output = layer([inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]])
    if mode == "ibp":
        output = layer([inputs[3], inputs[6]])

    f_pooling = K.function(inputs, output)

    y_ = f_ref(inputs_)
    z_ = z
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_pooling(inputs_)
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_pooling(inputs_)
    if mode == "ibp":
        u_c_, l_c_ = f_pooling(inputs_)

    assert_output_properties_box(
        x,
        y_,
        None,
        None,
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


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, floatx",
    [("channels_last", 0, 0, 1, 32), ("channels_last", 0, 0, 1, 64), ("channels_last", 0, 0, 1, 16)],
)
def test_MaxPooling2D_to_monotonic(data_format, odd, m_0, m_1, floatx):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    layer_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", dtype=K.floatx())
    output_ref = layer_ref(inputs[1])
    f_ref = K.function(inputs, output_ref)

    input_dim = x.shape[-1]
    layer = to_monotonic(layer_ref, input_dim=(2, input_dim), dc_decomp=True, fast=False)[0]
    output = layer(inputs[2:])

    f_pooling = K.function(inputs, output)

    y_ = f_ref(inputs_)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_pooling(inputs_)

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
        "maxpooling_{}".format(odd),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize("data_format, odd, m_0, m_1, fast", [("channels_last", 0, 0, 1, True)])
def test_MaxPooling2D_to_monotonic_nodc(data_format, odd, m_0, m_1, fast):

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    layer_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", dtype=K.floatx())
    output_ref = layer_ref(inputs[1])
    f_ref = K.function(inputs, output_ref)

    input_dim = x.shape[-1]
    layer = to_monotonic(layer_ref, input_dim=(2, input_dim), dc_decomp=False)[0]
    output = layer(inputs[2:])

    f_pooling = K.function(inputs, output)

    y_ = f_ref(inputs_)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_pooling(inputs_)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")
"""
