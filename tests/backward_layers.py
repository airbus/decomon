# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
import tensorflow.python.keras.backend as K
from decomon.backward_layers.backward_layers import get_backward
from decomon.layers.decomon_layers import DecomonActivation, DecomonPermute, DecomonReshape, DecomonFlatten
from tensorflow.keras.layers import Input, Activation, Flatten, Reshape
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_tensor_decomposition_multid_box,
)
from tensorflow.keras.models import Model

# Activation

"""
@pytest.mark.parametrize(
    "n, activation, mode, previous, floatx",
    [
        (0, "relu", "hybrid", True, 32),
        (1, "relu", "hybrid", True, 32),
        (2, "relu", "hybrid", True, 32),
        (3, "relu", "hybrid", True, 32),
        (4, "relu", "hybrid", True, 32),
        (5, "relu", "hybrid", True, 32),
        (6, "relu", "hybrid", True, 32),
        (0, "relu", "forward", True, 32),
        (1, "relu", "forward", True, 32),
        (2, "relu", "forward", True, 32),
        (3, "relu", "forward", True, 32),
        (4, "relu", "forward", True, 32),
        (5, "relu", "forward", True, 32),
        (6, "relu", "forward", True, 32),
        (0, "relu", "ibp", True, 32),
        (1, "relu", "ibp", True, 32),
        (2, "relu", "ibp", True, 32),
        (3, "relu", "ibp", True, 32),
        (4, "relu", "ibp", True, 32),
        (5, "relu", "ibp", True, 32),
        (6, "relu", "ibp", True, 32),
        (0, "relu", "hybrid", True, 64),
        (1, "relu", "hybrid", True, 64),
        (2, "relu", "hybrid", True, 64),
        (3, "relu", "hybrid", True, 64),
        (4, "relu", "hybrid", True, 64),
        (5, "relu", "hybrid", True, 64),
        (6, "relu", "hybrid", True, 64),
        (0, "relu", "forward", True, 64),
        (1, "relu", "forward", True, 64),
        (2, "relu", "forward", True, 64),
        (3, "relu", "forward", True, 64),
        (4, "relu", "forward", True, 64),
        (5, "relu", "forward", True, 64),
        (6, "relu", "forward", True, 64),
        (0, "relu", "ibp", True, 64),
        (1, "relu", "ibp", True, 64),
        (2, "relu", "ibp", True, 64),
        (3, "relu", "ibp", True, 64),
        (4, "relu", "ibp", True, 64),
        (5, "relu", "ibp", True, 64),
        (6, "relu", "ibp", True, 64),
        (0, "relu", "hybrid", True, 16),
        (1, "relu", "hybrid", True, 16),
        (2, "relu", "hybrid", True, 16),
        (3, "relu", "hybrid", True, 16),
        (4, "relu", "hybrid", True, 16),
        (5, "relu", "hybrid", True, 16),
        (6, "relu", "hybrid", True, 16),
        (0, "relu", "forward", True, 16),
        (1, "relu", "forward", True, 16),
        (2, "relu", "forward", True, 16),
        (3, "relu", "forward", True, 16),
        (4, "relu", "forward", True, 16),
        (5, "relu", "forward", True, 16),
        (6, "relu", "forward", True, 16),
        (0, "relu", "ibp", True, 16),
        (1, "relu", "ibp", True, 16),
        (2, "relu", "ibp", True, 16),
        (3, "relu", "ibp", True, 16),
        (4, "relu", "ibp", True, 16),
        (5, "relu", "ibp", True, 16),
        (6, "relu", "ibp", True, 16),
        (0, "relu", "hybrid", False, 32),
        (1, "relu", "hybrid", False, 32),
        (2, "relu", "hybrid", False, 32),
        (3, "relu", "hybrid", False, 32),
        (4, "relu", "hybrid", False, 32),
        (5, "relu", "hybrid", False, 32),
        (6, "relu", "hybrid", False, 32),
        (0, "relu", "forward", False, 32),
        (1, "relu", "forward", False, 32),
        (2, "relu", "forward", False, 32),
        (3, "relu", "forward", False, 32),
        (4, "relu", "forward", False, 32),
        (5, "relu", "forward", False, 32),
        (6, "relu", "forward", False, 32),
        (0, "relu", "ibp", False, 32),
        (1, "relu", "ibp", False, 32),
        (2, "relu", "ibp", False, 32),
        (3, "relu", "ibp", False, 32),
        (4, "relu", "ibp", False, 32),
        (5, "relu", "ibp", False, 32),
        (6, "relu", "ibp", False, 32),
        (0, "relu", "hybrid", False, 64),
        (1, "relu", "hybrid", False, 64),
        (2, "relu", "hybrid", False, 64),
        (3, "relu", "hybrid", False, 64),
        (4, "relu", "hybrid", False, 64),
        (5, "relu", "hybrid", False, 64),
        (6, "relu", "hybrid", False, 64),
        (0, "relu", "forward", False, 64),
        (1, "relu", "forward", False, 64),
        (2, "relu", "forward", False, 64),
        (3, "relu", "forward", False, 64),
        (4, "relu", "forward", False, 64),
        (5, "relu", "forward", False, 64),
        (6, "relu", "forward", False, 64),
        (0, "relu", "ibp", False, 64),
        (1, "relu", "ibp", False, 64),
        (2, "relu", "ibp", False, 64),
        (3, "relu", "ibp", False, 64),
        (4, "relu", "ibp", False, 64),
        (5, "relu", "ibp", False, 64),
        (6, "relu", "ibp", False, 64),
        (0, "relu", "hybrid", False, 16),
        (1, "relu", "hybrid", False, 16),
        (2, "relu", "hybrid", False, 16),
        (3, "relu", "hybrid", False, 16),
        (4, "relu", "hybrid", False, 16),
        (5, "relu", "hybrid", False, 16),
        (6, "relu", "hybrid", False, 16),
        (0, "relu", "forward", False, 16),
        (1, "relu", "forward", False, 16),
        (2, "relu", "forward", False, 16),
        (3, "relu", "forward", False, 16),
        (4, "relu", "forward", False, 16),
        (5, "relu", "forward", False, 16),
        (6, "relu", "forward", False, 16),
        (0, "relu", "ibp", False, 16),
        (1, "relu", "ibp", False, 16),
        (2, "relu", "ibp", False, 16),
        (3, "relu", "ibp", False, 16),
        (4, "relu", "ibp", False, 16),
        (5, "relu", "ibp", False, 16),
        (6, "relu", "ibp", False, 16),
        (0, "linear", "hybrid", True, 32),
        (1, "linear", "hybrid", True, 32),
        (2, "linear", "hybrid", True, 32),
        (3, "linear", "hybrid", True, 32),
        (4, "linear", "hybrid", True, 32),
        (5, "linear", "hybrid", True, 32),
        (6, "linear", "hybrid", True, 32),
        (0, "linear", "forward", True, 32),
        (1, "linear", "forward", True, 32),
        (2, "linear", "forward", True, 32),
        (3, "linear", "forward", True, 32),
        (4, "linear", "forward", True, 32),
        (5, "linear", "forward", True, 32),
        (6, "linear", "forward", True, 32),
        (0, "linear", "ibp", True, 32),
        (1, "linear", "ibp", True, 32),
        (2, "linear", "ibp", True, 32),
        (3, "linear", "ibp", True, 32),
        (4, "linear", "ibp", True, 32),
        (5, "linear", "ibp", True, 32),
        (6, "linear", "ibp", True, 32),
        (0, None, "hybrid", True, 32),
        (1, None, "hybrid", True, 32),
        (2, None, "hybrid", True, 32),
        (3, None, "hybrid", True, 32),
        (4, None, "hybrid", True, 32),
        (5, None, "hybrid", True, 32),
        (6, None, "hybrid", True, 32),
        (0, None, "forward", True, 32),
        (1, None, "forward", True, 32),
        (2, None, "forward", True, 32),
        (3, None, "forward", True, 32),
        (4, None, "forward", True, 32),
        (5, None, "forward", True, 32),
        (6, None, "forward", True, 32),
        (0, None, "ibp", True, 32),
        (1, None, "ibp", True, 32),
        (2, None, "ibp", True, 32),
        (3, None, "ibp", True, 32),
        (4, None, "ibp", True, 32),
        (5, None, "ibp", True, 32),
        (6, None, "ibp", True, 32),
    ],
)
def test_Backward_Activation_1D_box_model(n, activation, mode, previous, floatx):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonActivation(activation, dc_decomp=False, mode=mode)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    if mode == "hybrid":
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode == "forward":
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    if mode == "ibp":
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output

    w_out = Input((1, 1))
    b_out = Input((1,))
    # get backward layer
    layer_backward = get_backward(layer, previous=previous)
    if previous:
        w_out_u_, b_out_u_, w_out_l_, b_out_l_ = layer_backward(input_mode + [w_out, b_out, w_out, b_out])
        model = Model(inputs[2:] + [w_out, b_out], [w_out_u_, b_out_u_, w_out_l_, b_out_l_])
        w_u_, b_u_, w_l_, b_l_ = model.predict(
            inputs_[2:]
            + [
                np.ones((len(x), 1, 1)),
                np.zeros(
                    (
                        len(x),
                        1,
                    )
                ),
            ]
        )
    else:
        w_out_u_, b_out_u_, w_out_l_, b_out_l_ = layer_backward(input_mode)
        model = Model(inputs[2:], [w_out_u_, b_out_u_, w_out_l_, b_out_l_])
        w_u_, b_u_, w_l_, b_l_ = model.predict(inputs_[2:])

    assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_, 0) * W_u + np.minimum(w_u_, 0) * W_l, 1)[:, :, None],
        b_u_ + np.sum(np.maximum(w_u_, 0) * b_u[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_, 0) * W_l + np.minimum(w_l_, 0) * W_u, 1)[:, :, None],
        b_l_ + np.sum(np.maximum(w_l_, 0) * b_l[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * b_u[:, :, None], 1),
        "activation_{}".format(activation),
        decimal=decimal,
    )

    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, activation, floatx, mode, previous",
    [
        (0, None, 32, "hybrid", True),
        (1, None, 32, "hybrid", True),
        (0, "linear", 32, "hybrid", True),
        (1, "linear", 32, "hybrid", True),
        (0, "relu", 32, "hybrid", True),
        (0, None, 32, "forward", True),
        (1, None, 32, "forward", True),
        (0, "linear", 32, "forward", True),
        (1, "linear", 32, "forward", True),
        (0, "relu", 32, "forward", True),
        (0, None, 32, "ibp", True),
        (1, None, 32, "ibp", True),
        (0, "linear", 32, "ibp", True),
        (1, "linear", 32, "ibp", True),
        (0, "relu", 32, "ibp", True),
        (0, None, 64, "hybrid", True),
        (1, None, 64, "hybrid", True),
        (0, "linear", 64, "hybrid", True),
        (1, "linear", 64, "hybrid", True),
        (0, "relu", 64, "hybrid", True),
        (0, None, 64, "forward", True),
        (1, None, 64, "forward", True),
        (0, "linear", 64, "forward", True),
        (1, "linear", 64, "forward", True),
        (0, "relu", 64, "forward", True),
        (0, None, 64, "ibp", True),
        (1, None, 64, "ibp", True),
        (0, "linear", 64, "ibp", True),
        (1, "linear", 64, "ibp", True),
        (0, "relu", 64, "ibp", True),
        (0, None, 16, "hybrid", True),
        (1, None, 16, "hybrid", True),
        (0, "linear", 16, "hybrid", True),
        (1, "linear", 16, "hybrid", True),
        (0, "relu", 16, "hybrid", True),
        (0, None, 16, "forward", True),
        (1, None, 16, "forward", True),
        (0, "linear", 16, "forward", True),
        (1, "linear", 16, "forward", True),
        (0, "relu", 16, "forward", True),
        (0, None, 16, "ibp", True),
        (1, None, 16, "ibp", True),
        (0, "linear", 16, "ibp", True),
        (1, "linear", 16, "ibp", True),
        (0, "relu", 16, "ibp", True),
        (0, None, 32, "hybrid", False),
        (1, None, 32, "hybrid", False),
        (0, "linear", 32, "hybrid", False),
        (1, "linear", 32, "hybrid", False),
        (0, "relu", 32, "hybrid", False),
        (0, None, 32, "forward", False),
        (1, None, 32, "forward", False),
        (0, "linear", 32, "forward", False),
        (1, "linear", 32, "forward", False),
        (0, "relu", 32, "forward", False),
        (0, None, 32, "ibp", False),
        (1, None, 32, "ibp", False),
        (0, "linear", 32, "ibp", False),
        (1, "linear", 32, "ibp", False),
        (0, "relu", 32, "ibp", False),
        (0, None, 64, "hybrid", False),
        (1, None, 64, "hybrid", False),
        (0, "linear", 64, "hybrid", False),
        (1, "linear", 64, "hybrid", False),
        (0, "relu", 64, "hybrid", False),
        (0, None, 64, "forward", False),
        (1, None, 64, "forward", False),
        (0, "linear", 64, "forward", False),
        (1, "linear", 64, "forward", False),
        (0, "relu", 64, "forward", False),
        (0, None, 64, "ibp", False),
        (1, None, 64, "ibp", False),
        (0, "linear", 64, "ibp", False),
        (1, "linear", 64, "ibp", False),
        (0, "relu", 64, "ibp", False),
        (0, None, 16, "hybrid", False),
        (1, None, 16, "hybrid", False),
        (0, "linear", 16, "hybrid", False),
        (1, "linear", 16, "hybrid", False),
        (0, "relu", 16, "hybrid", False),
        (0, None, 16, "forward", False),
        (1, None, 16, "forward", False),
        (0, "linear", 16, "forward", False),
        (1, "linear", 16, "forward", False),
        (0, "relu", 16, "forward", False),
        (0, None, 16, "ibp", False),
        (1, None, 16, "ibp", False),
        (0, "linear", 16, "ibp", False),
        (1, "linear", 16, "ibp", False),
        (0, "relu", 16, "ibp", False),
    ],
)
def test_Backward_Activation_multiD_box(odd, activation, floatx, mode, previous):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonActivation(activation, dc_decomp=False, mode=mode)
    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    if mode == "hybrid":
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode == "forward":
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    if mode == "ibp":
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output

    # output = layer(inputs[2:])
    # z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1))
    b_out = Input((1,))
    # get backward layer
    layer_backward = get_backward(layer, previous=previous)
    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode + [w_out, b_out, w_out, b_out])
        f_dense = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_dense(
            inputs_
            + [
                np.ones((len(x), 1, 1)),
                np.zeros(
                    (
                        len(x),
                        1,
                    )
                ),
            ]
        )
    else:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        # import pdb; pdb.set_trace()
        output_ = f_dense(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_, 0) * W_u + np.minimum(w_u_, 0) * W_l, 1)[:, :, None],
        b_u_ + np.sum(np.maximum(w_u_, 0) * b_u[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_, 0) * W_l + np.minimum(w_l_, 0) * W_u, 1)[:, :, None],
        b_l_ + np.sum(np.maximum(w_l_, 0) * b_l[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * b_u[:, :, None], 1),
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, floatx, mode, previous, data_format",
    [
        (0, 32, "hybrid", True, "channels_last"),
        (1, 32, "hybrid", True, "channels_last"),
        (0, 32, "forward", True, "channels_last"),
        (1, 32, "forward", True, "channels_last"),
        (0, 32, "ibp", True, "channels_last"),
        (1, 32, "ibp", True, "channels_last"),
        (0, 32, "hybrid", True, "channels_first"),
        (1, 32, "hybrid", True, "channels_first"),
        (0, 32, "forward", True, "channels_first"),
        (1, 32, "forward", True, "channels_first"),
        (0, 32, "ibp", True, "channels_first"),
        (1, 32, "ibp", True, "channels_first"),
        (0, 32, "hybrid", False, "channels_last"),
        (1, 32, "hybrid", False, "channels_last"),
        (0, 32, "forward", False, "channels_last"),
        (1, 32, "forward", False, "channels_last"),
        (0, 32, "ibp", False, "channels_last"),
        (1, 32, "ibp", False, "channels_last"),
        (0, 32, "hybrid", False, "channels_first"),
        (1, 32, "hybrid", False, "channels_first"),
        (0, 32, "forward", False, "channels_first"),
        (1, 32, "forward", False, "channels_first"),
        (0, 32, "ibp", False, "channels_first"),
        (1, 32, "ibp", False, "channels_first"),
        (0, 64, "hybrid", True, "channels_last"),
        (1, 64, "hybrid", True, "channels_last"),
        (0, 64, "forward", True, "channels_last"),
        (1, 64, "forward", True, "channels_last"),
        (0, 64, "ibp", True, "channels_last"),
        (1, 64, "ibp", True, "channels_last"),
        (0, 64, "hybrid", True, "channels_first"),
        (1, 64, "hybrid", True, "channels_first"),
        (0, 64, "forward", True, "channels_first"),
        (1, 64, "forward", True, "channels_first"),
        (0, 64, "ibp", True, "channels_first"),
        (1, 64, "ibp", True, "channels_first"),
        (0, 64, "hybrid", False, "channels_last"),
        (1, 64, "hybrid", False, "channels_last"),
        (0, 64, "forward", False, "channels_last"),
        (1, 64, "forward", False, "channels_last"),
        (0, 64, "ibp", False, "channels_last"),
        (1, 64, "ibp", False, "channels_last"),
        (0, 64, "hybrid", False, "channels_first"),
        (1, 64, "hybrid", False, "channels_first"),
        (0, 64, "forward", False, "channels_first"),
        (1, 64, "forward", False, "channels_first"),
        (0, 64, "ibp", False, "channels_first"),
        (1, 64, "ibp", False, "channels_first"),
        (0, 16, "hybrid", True, "channels_last"),
        (1, 16, "hybrid", True, "channels_last"),
        (0, 16, "forward", True, "channels_last"),
        (1, 16, "forward", True, "channels_last"),
        (0, 16, "ibp", True, "channels_last"),
        (1, 16, "ibp", True, "channels_last"),
        (0, 16, "hybrid", True, "channels_first"),
        (1, 16, "hybrid", True, "channels_first"),
        (0, 16, "forward", True, "channels_first"),
        (1, 16, "forward", True, "channels_first"),
        (0, 16, "ibp", True, "channels_first"),
        (1, 16, "ibp", True, "channels_first"),
        (0, 16, "hybrid", False, "channels_last"),
        (1, 16, "hybrid", False, "channels_last"),
        (0, 16, "forward", False, "channels_last"),
        (1, 16, "forward", False, "channels_last"),
        (0, 16, "ibp", False, "channels_last"),
        (1, 16, "ibp", False, "channels_last"),
        (0, 16, "hybrid", False, "channels_first"),
        (1, 16, "hybrid", False, "channels_first"),
        (0, 16, "forward", False, "channels_first"),
        (1, 16, "forward", False, "channels_first"),
        (0, 16, "ibp", False, "channels_first"),
        (1, 16, "ibp", False, "channels_first"),
    ],
)
def test_Backward_Flatten_multiD_box(odd, floatx, mode, previous, data_format):

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonFlatten("channels_last", dc_decomp=False, mode=mode)
    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    if mode == "hybrid":
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode == "forward":
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    if mode == "ibp":
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output

    # output = layer(inputs[2:])
    # z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1))
    b_out = Input((1,))
    # get backward layer
    layer_backward = get_backward(layer, previous=previous)
    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode + [w_out, b_out, w_out, b_out])
        f_dense = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_dense(
            inputs_
            + [
                np.ones((len(x), 1, 1)),
                np.zeros(
                    (
                        len(x),
                        1,
                    )
                ),
            ]
        )
    else:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        # import pdb; pdb.set_trace()
        output_ = f_dense(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_, 0) * W_u + np.minimum(w_u_, 0) * W_l, 1)[:, :, None],
        b_u_ + np.sum(np.maximum(w_u_, 0) * b_u[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_, 0) * W_l + np.minimum(w_l_, 0) * W_u, 1)[:, :, None],
        b_l_ + np.sum(np.maximum(w_l_, 0) * b_l[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * b_u[:, :, None], 1),
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, floatx, mode, previous, data_format",
    [
        (0, 32, "hybrid", True, "channels_last"),
        (1, 32, "hybrid", True, "channels_last"),
        (0, 32, "forward", True, "channels_last"),
        (1, 32, "forward", True, "channels_last"),
        (0, 32, "ibp", True, "channels_last"),
        (1, 32, "ibp", True, "channels_last"),
        (0, 32, "hybrid", True, "channels_first"),
        (1, 32, "hybrid", True, "channels_first"),
        (0, 32, "forward", True, "channels_first"),
        (1, 32, "forward", True, "channels_first"),
        (0, 32, "ibp", True, "channels_first"),
        (1, 32, "ibp", True, "channels_first"),
        (0, 32, "hybrid", False, "channels_last"),
        (1, 32, "hybrid", False, "channels_last"),
        (0, 32, "forward", False, "channels_last"),
        (1, 32, "forward", False, "channels_last"),
        (0, 32, "ibp", False, "channels_last"),
        (1, 32, "ibp", False, "channels_last"),
        (0, 32, "hybrid", False, "channels_first"),
        (1, 32, "hybrid", False, "channels_first"),
        (0, 32, "forward", False, "channels_first"),
        (1, 32, "forward", False, "channels_first"),
        (0, 32, "ibp", False, "channels_first"),
        (1, 32, "ibp", False, "channels_first"),
        (0, 64, "hybrid", True, "channels_last"),
        (1, 64, "hybrid", True, "channels_last"),
        (0, 64, "forward", True, "channels_last"),
        (1, 64, "forward", True, "channels_last"),
        (0, 64, "ibp", True, "channels_last"),
        (1, 64, "ibp", True, "channels_last"),
        (0, 64, "hybrid", True, "channels_first"),
        (1, 64, "hybrid", True, "channels_first"),
        (0, 64, "forward", True, "channels_first"),
        (1, 64, "forward", True, "channels_first"),
        (0, 64, "ibp", True, "channels_first"),
        (1, 64, "ibp", True, "channels_first"),
        (0, 64, "hybrid", False, "channels_last"),
        (1, 64, "hybrid", False, "channels_last"),
        (0, 64, "forward", False, "channels_last"),
        (1, 64, "forward", False, "channels_last"),
        (0, 64, "ibp", False, "channels_last"),
        (1, 64, "ibp", False, "channels_last"),
        (0, 64, "hybrid", False, "channels_first"),
        (1, 64, "hybrid", False, "channels_first"),
        (0, 64, "forward", False, "channels_first"),
        (1, 64, "forward", False, "channels_first"),
        (0, 64, "ibp", False, "channels_first"),
        (1, 64, "ibp", False, "channels_first"),
        (0, 16, "hybrid", True, "channels_last"),
        (1, 16, "hybrid", True, "channels_last"),
        (0, 16, "forward", True, "channels_last"),
        (1, 16, "forward", True, "channels_last"),
        (0, 16, "ibp", True, "channels_last"),
        (1, 16, "ibp", True, "channels_last"),
        (0, 16, "hybrid", True, "channels_first"),
        (1, 16, "hybrid", True, "channels_first"),
        (0, 16, "forward", True, "channels_first"),
        (1, 16, "forward", True, "channels_first"),
        (0, 16, "ibp", True, "channels_first"),
        (1, 16, "ibp", True, "channels_first"),
        (0, 16, "hybrid", False, "channels_last"),
        (1, 16, "hybrid", False, "channels_last"),
        (0, 16, "forward", False, "channels_last"),
        (1, 16, "forward", False, "channels_last"),
        (0, 16, "ibp", False, "channels_last"),
        (1, 16, "ibp", False, "channels_last"),
        (0, 16, "hybrid", False, "channels_first"),
        (1, 16, "hybrid", False, "channels_first"),
        (0, 16, "forward", False, "channels_first"),
        (1, 16, "forward", False, "channels_first"),
        (0, 16, "ibp", False, "channels_first"),
        (1, 16, "ibp", False, "channels_first"),
    ],
)
def test_Backward_Reshape_multiD_box(odd, floatx, mode, previous, data_format):
    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonReshape((-1,), dc_decomp=False, mode=mode)
    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    if mode == "hybrid":
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode == "forward":
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    if mode == "ibp":
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output

    # output = layer(inputs[2:])
    # z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1))
    b_out = Input((1,))
    # get backward layer
    layer_backward = get_backward(layer, previous=previous)
    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode + [w_out, b_out, w_out, b_out])
        f_dense = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_dense(
            inputs_
            + [
                np.ones((len(x), 1, 1)),
                np.zeros(
                    (
                        len(x),
                        1,
                    )
                ),
            ]
        )
    else:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        # import pdb; pdb.set_trace()
        output_ = f_dense(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_, 0) * W_u + np.minimum(w_u_, 0) * W_l, 1)[:, :, None],
        b_u_ + np.sum(np.maximum(w_u_, 0) * b_u[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_, 0) * W_l + np.minimum(w_l_, 0) * W_u, 1)[:, :, None],
        b_l_ + np.sum(np.maximum(w_l_, 0) * b_l[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * b_u[:, :, None], 1),
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)
"""


@pytest.mark.parametrize(
    "odd, floatx, mode, previous, data_format",
    [
        (0, 32, "hybrid", True, "channels_last"), (1, 32, "hybrid", True, "channels_last"),
        (0, 32, "forward", True, "channels_last"), (1, 32, "forward", True, "channels_last"),
        (0, 32, "ibp", True, "channels_last"), (1, 32, "ibp", True, "channels_last"),
        (0, 32, "hybrid", True, "channels_first"), (1, 32, "hybrid", True, "channels_first"),
        (0, 32, "forward", True, "channels_first"), (1, 32, "forward", True, "channels_first"),
        (0, 32, "ibp", True, "channels_first"), (1, 32, "ibp", True, "channels_first"),
        (0, 32, "hybrid", False, "channels_last"), (1, 32, "hybrid", False, "channels_last"),
        (0, 32, "forward", False, "channels_last"), (1, 32, "forward", False, "channels_last"),
        (0, 32, "ibp", False, "channels_last"), (1, 32, "ibp", False, "channels_last"),
        (0, 32, "hybrid", False, "channels_first"), (1, 32, "hybrid", False, "channels_first"),
        (0, 32, "forward", False, "channels_first"), (1, 32, "forward", False, "channels_first"),
        (0, 32, "ibp", False, "channels_first"), (1, 32, "ibp", False, "channels_first"),
        (0, 64, "hybrid", True, "channels_last"), (1, 64, "hybrid", True, "channels_last"),
        (0, 64, "forward", True, "channels_last"), (1, 64, "forward", True, "channels_last"),
        (0, 64, "ibp", True, "channels_last"), (1, 64, "ibp", True, "channels_last"),
        (0, 64, "hybrid", True, "channels_first"), (1, 64, "hybrid", True, "channels_first"),
        (0, 64, "forward", True, "channels_first"), (1, 64, "forward", True, "channels_first"),
        (0, 64, "ibp", True, "channels_first"), (1, 64, "ibp", True, "channels_first"),
        (0, 64, "hybrid", False, "channels_last"), (1, 64, "hybrid", False, "channels_last"),
        (0, 64, "forward", False, "channels_last"), (1, 64, "forward", False, "channels_last"),
        (0, 64, "ibp", False, "channels_last"), (1, 64, "ibp", False, "channels_last"),
        (0, 64, "hybrid", False, "channels_first"), (1, 64, "hybrid", False, "channels_first"),
        (0, 64, "forward", False, "channels_first"), (1, 64, "forward", False, "channels_first"),
        (0, 64, "ibp", False, "channels_first"), (1, 64, "ibp", False, "channels_first"),
        (0, 16, "hybrid", True, "channels_last"), (1, 16, "hybrid", True, "channels_last"),
        (0, 16, "forward", True, "channels_last"), (1, 16, "forward", True, "channels_last"),
        (0, 16, "ibp", True, "channels_last"), (1, 16, "ibp", True, "channels_last"),
        (0, 16, "hybrid", True, "channels_first"), (1, 16, "hybrid", True, "channels_first"),
        (0, 16, "forward", True, "channels_first"), (1, 16, "forward", True, "channels_first"),
        (0, 16, "ibp", True, "channels_first"), (1, 16, "ibp", True, "channels_first"),
        (0, 16, "hybrid", False, "channels_last"), (1, 16, "hybrid", False, "channels_last"),
        (0, 16, "forward", False, "channels_last"), (1, 16, "forward", False, "channels_last"),
        (0, 16, "ibp", False, "channels_last"), (1, 16, "ibp", False, "channels_last"),
        (0, 16, "hybrid", False, "channels_first"), (1, 16, "hybrid", False, "channels_first"),
        (0, 16, "forward", False, "channels_first"), (1, 16, "forward", False, "channels_first"),
        (0, 16, "ibp", False, "channels_first"), (1, 16, "ibp", False, "channels_first"),

    ]
)
def test_Backward_Permute_multiD_box(odd, floatx, mode, previous, data_format):
    if data_format == 'channels_first' and not len(K._get_available_gpus()):
        return

    K.set_floatx('float{}'.format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2


    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    n_dim = len(y.shape) - 1
    target_shape = np.random.permutation(n_dim) + 1
    target_shape_ = tuple([0] + list(target_shape))
    layer = DecomonPermute(target_shape, dc_decomp=False, mode=mode)


    if mode == "hybrid":
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode == "forward":
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    if mode == "ibp":
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output

    # output = layer(inputs[2:])
    # z_0, u_c_0, _, _, l_c_0, _, _ = output
    n_out = y.shape[-1]
    w_out = Input((n_out, n_out))
    b_out = Input((n_out,))

    w_out_ = np.concatenate([np.diag([1]*n_out)[None]]*len(x))
    b_out_ = np.zeros((len(x), n_out))
    # get backward layer
    layer_backward = get_backward(layer, previous=previous)
    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode + [w_out, b_out, w_out, b_out])
        f_dense = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])

        output_ = f_dense(inputs_+[w_out_, b_out_])
        #output_ = f_dense(inputs_ + [np.ones((len(x), 1, 1)), np.zeros((len(x), 1,))])
    else:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        # import pdb; pdb.set_trace()
        output_ = f_dense(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_
    
    assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_, 0) * W_u + np.minimum(w_u_, 0) * W_l, 1)[:, :, None],
        b_u_ + np.sum(np.maximum(w_u_, 0) * b_u[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_, 0) * W_l + np.minimum(w_l_, 0) * W_u, 1)[:, :, None],
        b_l_
        + np.sum(np.maximum(w_l_, 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_, 0) * b_u[:, :, None], 1),
        "dense_{}".format(odd),
        decimal=decimal
    )
    

    K.set_floatx("float32")
    K.set_epsilon(eps)

