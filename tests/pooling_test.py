# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from . import (
    assert_output_properties_box,
    get_tensor_decomposition_images_box,
    get_standard_values_images_box,
)
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import MaxPooling2D
from decomon.layers.decomon_layers import to_monotonic

from decomon.layers.maxpooling import DecomonMaxPooling2D


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_MaxPooling2D_box(data_format, odd, m_0, m_1):

    inputs = get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_
    output_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(inputs[1])
    f_ref = K.function(inputs, output_ref)

    layer = DecomonMaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", dc_decomp=True)

    output = layer(inputs[1:])

    f_pooling = K.function(inputs, output)

    y_ref = f_ref(inputs_)
    z_ = f_pooling(inputs_)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_pooling(inputs_)

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
        "maxpooling_{}".format(odd),
    )


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_MaxPooling2D_box_nodc(data_format, odd, m_0, m_1):

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    output_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(inputs[1])
    f_ref = K.function(inputs, output_ref)

    layer = DecomonMaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", dc_decomp=False)

    output = layer(inputs[1:])

    f_pooling = K.function(inputs, output)

    y_ref = f_ref(inputs_)
    z_ = f_pooling(inputs_)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_pooling(inputs_)

    assert np.allclose(y_, y_ref)


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_MaxPooling2D_to_monotonic(data_format, odd, m_0, m_1):

    inputs = get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    layer_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
    output_ref = layer_ref(inputs[1])
    f_ref = K.function(inputs, output_ref)

    input_dim = x.shape[-1]
    layer = to_monotonic(layer_ref, input_dim=(2, input_dim), dc_decomp=True)
    output = layer(inputs[1:])

    f_pooling = K.function(inputs, output)

    y_ref = f_ref(inputs_)
    z_ = f_pooling(inputs_)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_pooling(inputs_)

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
        "maxpooling_{}".format(odd),
    )


@pytest.mark.parametrize("data_format, odd, m_0, m_1", [("channels_last", 0, 0, 1)])
def test_MaxPooling2D_to_monotonic_nodc(data_format, odd, m_0, m_1):

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    layer_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
    output_ref = layer_ref(inputs[1])
    f_ref = K.function(inputs, output_ref)

    input_dim = x.shape[-1]
    layer = to_monotonic(layer_ref, input_dim=(2, input_dim), dc_decomp=False)
    output = layer(inputs[1:])

    f_pooling = K.function(inputs, output)

    y_ref = f_ref(inputs_)
    z_ = f_pooling(inputs_)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_pooling(inputs_)

    assert np.allclose(y_, y_ref)
