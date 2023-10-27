# Test unit for decomon with Dense layers


import keras.config as keras_config
import numpy as np
import pytest
from tensorflow.python.keras.backend import _get_available_gpus

from decomon.backward_layers.convert import to_backward
from decomon.layers.decomon_layers import DecomonConv2D


def test_Decomon_conv_box(data_format, padding, use_bias, mode, floatx, decimal, helpers):
    if data_format == "channels_first" and not helpers.tensorflow_in_GPU_mode():
        pytest.skip("data format 'channels first' is possible only in GPU mode")

    odd, m_0, m_1 = 0, 0, 1
    dc_decomp = False

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    # decomon layer
    decomon_layer = DecomonConv2D(
        10,
        kernel_size=(3, 3),
        dc_decomp=dc_decomp,
        padding=padding,
        use_bias=use_bias,
        mode=mode,
        dtype=keras_config.floatx(),
    )
    decomon_layer(inputs_for_mode)  # init weights

    # get backward layer
    backward_layer = to_backward(decomon_layer)

    # backward outputs
    outputs = backward_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    w_u_, b_u_, w_l_, b_l_ = outputs_

    # reshape the matrices
    w_u_ = w_u_[:, None]
    w_l_ = w_l_[:, None]
    b_u_ = b_u_[:, None]
    b_l_ = b_l_[:, None]

    b_l = b_l.reshape((len(b_l), -1))
    b_u = b_u.reshape((len(b_u), -1))
    W_u = W_u.reshape((len(W_u), W_u.shape[1], -1))
    W_l = W_l.reshape((len(W_l), W_l.shape[1], -1))

    # backward recomposition
    w_r_u = np.sum(np.maximum(0.0, w_u_) * np.expand_dims(W_u, -1), 2) + np.sum(
        np.minimum(0.0, w_u_) * np.expand_dims(W_l, -1), 2
    )
    w_r_l = np.sum(np.maximum(0.0, w_l_) * np.expand_dims(W_l, -1), 2) + np.sum(
        np.minimum(0.0, w_l_) * np.expand_dims(W_u, -1), 2
    )
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

    # check bounds consistency
    helpers.assert_output_properties_box_linear(x, None, z[:, 0], z[:, 1], None, w_r_u, b_r_u, None, w_r_l, b_r_l)
