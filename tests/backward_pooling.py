# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from . import (
    assert_output_properties_box,
    assert_output_properties_box_linear,
    get_tensor_decomposition_images_box,
    get_standard_values_images_box,
)
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Input
from decomon.layers.decomon_layers import to_monotonic

from decomon.layers.maxpooling import DecomonMaxPooling2D
from decomon.backward_layers.backward_layers import get_backward


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, fast", [("channels_last", 0, 0, 1, True), ("channels_last", 0, 0, 1, False)]
)
def test_backward_MaxPooling2D_box(data_format, odd, m_0, m_1, fast):

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # output_ref = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(inputs[1])

    layer = DecomonMaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", dc_decomp=False, fast=fast)
    layer_backward = get_backward(layer)

    output = layer(inputs[1:])
    n = np.prod(output[0].shape[1:])

    w_out = Input((1, n, n))
    b_out = Input((1, n))

    bounds = layer_backward(inputs[1:] + [w_out, b_out, w_out, b_out])

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[1:])

    f_pooling = K.function(inputs + [w_out, b_out], output + list(bounds))

    w_init = np.concatenate([np.diag([1.0] * n)[None]] * len(x)).reshape((-1, 1, n, n))
    b_init = np.zeros((len(x), 1, n))

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, w_u_0, b_u_0, w_l_0, b_l_0 = f_pooling(inputs_ + [w_init, b_init])

    W_u = W_u.reshape((len(W_u), W_u.shape[1], -1))[:, :, :, None]
    W_l = W_l.reshape((len(W_l), W_l.shape[1], -1))[:, :, :, None]
    b_u = b_u.reshape((len(b_u), 1, -1, 1))
    b_l = b_l.reshape((len(b_l), 1, -1, 1))

    w_u_1 = np.sum(W_u * np.maximum(w_u_0, 0), 2).reshape(w_u_.shape) + np.sum(W_l * np.minimum(w_u_0, 0), 2).reshape(
        w_u_.shape
    )
    w_l_1 = np.sum(W_l * np.maximum(w_l_0, 0), 2).reshape(w_u_.shape) + np.sum(W_u * np.minimum(w_l_0, 0), 2).reshape(
        w_u_.shape
    )

    b_u_1 = (np.sum(b_u * np.maximum(w_u_0, 0) + b_l * np.minimum(w_u_0, 0), 2) + b_u_0).reshape(b_u_.shape)
    b_l_1 = (np.sum(b_l * np.maximum(w_l_0, 0) + b_u * np.minimum(w_l_0, 0), 2) + b_l_0).reshape(b_l_.shape)

    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_1,
        b_u_1,
        l_c_,
        w_l_1,
        b_l_1,
        "dense_{}".format(n),
    )
