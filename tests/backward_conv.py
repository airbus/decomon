# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from . import (
    get_standard_values_images_box,
    get_tensor_decomposition_images_box,
    assert_output_properties_box_linear,
)
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Input
from decomon.layers.decomon_layers import DecomonConv2D
from decomon.backward_layers.backward_layers import get_backward


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, activation, padding",
    [
        ("channels_last", 0, 0, 1, None, "same"),
        ("channels_last", 0, 0, 1, "linear", "same"),
        ("channels_last", 0, 0, 1, "relu", "same"),
        ("channels_last", 0, 0, 1, None, "valid"),
        ("channels_last", 0, 0, 1, "linear", "valid"),
        ("channels_last", 0, 0, 1, "relu", "valid"),
    ],
)
def test_Decomon_conv_box(data_format, odd, m_0, m_1, activation, padding):

    layer = DecomonConv2D(10, kernel_size=(3, 3), activation=activation, dc_decomp=False, padding=padding)

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[1:])

    y_0, z_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = output

    output_shape = np.prod(y_0.shape[1:])

    w_out = Input((1, output_shape, output_shape))
    b_out = Input((1, output_shape))
    # get backward layer
    layer_backward = get_backward(layer)

    w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(inputs[1:] + [w_out, b_out, w_out, b_out])

    f_conv = K.function(inputs + [w_out, b_out], [y_0, z_0, u_c_0, w_out_u, b_out_u, l_c_0, w_out_l, b_out_l])

    w_init = np.concatenate([np.diag([1.0] * output_shape)[None]] * len(x)).reshape((-1, 1, output_shape, output_shape))
    b_init = np.zeros((len(x), 1, output_shape))
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    output_ = f_conv(inputs_ + [w_init, b_init])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_

    # step 1: flatten W_u
    W_u = W_u.reshape((len(W_u), W_u.shape[1], -1))
    W_l = W_l.reshape((len(W_l), W_l.shape[1], -1))

    w_r_u = np.sum(np.maximum(0.0, w_u_) * np.expand_dims(W_u, -1), 2) + np.sum(
        np.minimum(0.0, w_u_) * np.expand_dims(W_l, -1), 2
    )
    w_r_l = np.sum(np.maximum(0.0, w_l_) * np.expand_dims(W_l, -1), 2) + np.sum(
        np.minimum(0.0, w_l_) * np.expand_dims(W_u, -1), 2
    )

    b_l = b_l.reshape((len(b_l), -1))
    b_u = b_u.reshape((len(b_u), -1))

    b_r_u = (
        np.sum(np.maximum(0, w_u_[:, 0]) * np.expand_dims(b_u, -1), 1)[:, None]
        + np.sum(np.minimum(0, w_u_[:, 0]) * np.expand_dims(b_l, -1), 1)[:, None]
        + b_u_
    )
    b_r_l = (
        np.sum(np.maximum(0, w_l_[:, 0]) * np.expand_dims(b_l, -1), 1)[:, None]
        + np.sum(np.minimum(0, w_l_[:, 0]) * np.expand_dims(b_u, -1), 1)[:, None]
        + b_l_
    )

    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_r_u, b_r_u, l_c_, w_r_l, b_r_l, "nodc")
