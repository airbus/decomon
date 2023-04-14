# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.python.keras.backend import _get_available_gpus

from decomon.backward_layers.backward_layers import to_backward
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import DecomonConv2D


def test_Decomon_conv_box(data_format, padding, use_bias, mode, floatx, helpers):

    if data_format == "channels_first" and not len(_get_available_gpus()):
        return

    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonConv2D(
        10,
        kernel_size=(3, 3),
        dc_decomp=False,
        padding=padding,
        use_bias=use_bias,
        mode=mode,
        dtype=K.floatx(),
    )

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    elif mode == ForwardMode.AFFINE:
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    elif mode == ForwardMode.IBP:
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output
    else:
        raise ValueError("Unknown mode.")

    output_shape = np.prod(output[-1].shape[1:])

    w_out = Input((output_shape, output_shape))
    b_out = Input((output_shape,))
    # get backward layer
    layer_backward = to_backward(layer)

    w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
    f_conv = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
    output_ = f_conv(inputs_)

    w_u_, b_u_, w_l_, b_l_ = output_

    w_u_ = w_u_[:, None]
    w_l_ = w_l_[:, None]
    b_u_ = b_u_[:, None]
    b_l_ = b_l_[:, None]

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

    helpers.assert_output_properties_box_linear(
        x, None, z_[:, 0], z_[:, 1], None, w_r_u, b_r_u, None, w_r_l, b_r_l, "nodc"
    )

    K.set_floatx("float32")
    K.set_epsilon(eps)
