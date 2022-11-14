# Test unit for decomon with Dense layers
from __future__ import absolute_import

import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from decomon.backward_layers.backward_layers import get_backward
from decomon.layers.decomon_layers import DecomonDense, to_monotonic

from . import (
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_standart_values_1d_box,
    get_tensor_decomposition_1d_box,
    get_tensor_decomposition_multid_box,
)

###### native


@pytest.mark.parametrize(
    "n, activation, use_bias, slope, previous, mode, floatx",
    [
        (0, "linear", True, "volume-slope", True, "hybrid", 32),
        (1, "linear", True, "volume-slope", True, "hybrid", 32),
        (2, "linear", True, "volume-slope", True, "hybrid", 32),
        (3, "linear", True, "volume-slope", True, "hybrid", 32),
        (4, "linear", True, "volume-slope", True, "hybrid", 32),
        (5, "linear", True, "volume-slope", True, "hybrid", 32),
        (6, "linear", True, "volume-slope", True, "hybrid", 32),
        (7, "linear", True, "volume-slope", True, "hybrid", 32),
        (8, "linear", True, "volume-slope", True, "hybrid", 32),
        (9, "linear", True, "volume-slope", True, "hybrid", 32),
        (0, None, True, "volume-slope", True, "hybrid", 32),
        (1, None, True, "volume-slope", True, "hybrid", 32),
        (2, None, True, "volume-slope", True, "hybrid", 32),
        (3, None, True, "volume-slope", True, "hybrid", 32),
        (4, None, True, "volume-slope", True, "hybrid", 32),
        (5, None, True, "volume-slope", True, "hybrid", 32),
        (6, None, True, "volume-slope", True, "hybrid", 32),
        (7, None, True, "volume-slope", True, "hybrid", 32),
        (8, None, True, "volume-slope", True, "hybrid", 32),
        (9, None, True, "volume-slope", True, "hybrid", 32),
        (0, "linear", False, "volume-slope", True, "hybrid", 32),
        (1, "linear", False, "volume-slope", True, "hybrid", 32),
        (2, "linear", False, "volume-slope", True, "hybrid", 32),
        (3, "linear", False, "volume-slope", True, "hybrid", 32),
        (4, "linear", False, "volume-slope", True, "hybrid", 32),
        (5, "linear", False, "volume-slope", True, "hybrid", 32),
        (6, "linear", False, "volume-slope", True, "hybrid", 32),
        (7, "linear", False, "volume-slope", True, "hybrid", 32),
        (8, "linear", False, "volume-slope", True, "hybrid", 32),
        (9, "linear", False, "volume-slope", True, "hybrid", 32),
        (0, None, False, "volume-slope", True, "hybrid", 32),
        (1, None, False, "volume-slope", True, "hybrid", 32),
        (2, None, False, "volume-slope", True, "hybrid", 32),
        (3, None, False, "volume-slope", True, "hybrid", 32),
        (4, None, False, "volume-slope", True, "hybrid", 32),
        (5, None, False, "volume-slope", True, "hybrid", 32),
        (6, None, False, "volume-slope", True, "hybrid", 32),
        (7, None, False, "volume-slope", True, "hybrid", 32),
        (8, None, False, "volume-slope", True, "hybrid", 32),
        (9, None, False, "volume-slope", True, "hybrid", 32),
        (0, "linear", True, "volume-slope", False, "hybrid", 32),
        (1, "linear", True, "volume-slope", False, "hybrid", 32),
        (2, "linear", True, "volume-slope", False, "hybrid", 32),
        (3, "linear", True, "volume-slope", False, "hybrid", 32),
        (4, "linear", True, "volume-slope", False, "hybrid", 32),
        (5, "linear", True, "volume-slope", False, "hybrid", 32),
        (6, "linear", True, "volume-slope", False, "hybrid", 32),
        (7, "linear", True, "volume-slope", False, "hybrid", 32),
        (8, "linear", True, "volume-slope", False, "hybrid", 32),
        (9, "linear", True, "volume-slope", False, "hybrid", 32),
        (0, None, True, "volume-slope", False, "hybrid", 32),
        (1, None, True, "volume-slope", False, "hybrid", 32),
        (2, None, True, "volume-slope", False, "hybrid", 32),
        (3, None, True, "volume-slope", False, "hybrid", 32),
        (4, None, True, "volume-slope", False, "hybrid", 32),
        (5, None, True, "volume-slope", False, "hybrid", 32),
        (6, None, True, "volume-slope", False, "hybrid", 32),
        (7, None, True, "volume-slope", False, "hybrid", 32),
        (8, None, True, "volume-slope", False, "hybrid", 32),
        (9, None, True, "volume-slope", False, "hybrid", 32),
        (0, "linear", False, "volume-slope", False, "hybrid", 32),
        (1, "linear", False, "volume-slope", False, "hybrid", 32),
        (2, "linear", False, "volume-slope", False, "hybrid", 32),
        (3, "linear", False, "volume-slope", False, "hybrid", 32),
        (4, "linear", False, "volume-slope", False, "hybrid", 32),
        (5, "linear", False, "volume-slope", False, "hybrid", 32),
        (6, "linear", False, "volume-slope", False, "hybrid", 32),
        (7, "linear", False, "volume-slope", False, "hybrid", 32),
        (8, "linear", False, "volume-slope", False, "hybrid", 32),
        (9, "linear", False, "volume-slope", False, "hybrid", 32),
        (0, None, False, "volume-slope", False, "hybrid", 32),
        (1, None, False, "volume-slope", False, "hybrid", 32),
        (2, None, False, "volume-slope", False, "hybrid", 32),
        (3, None, False, "volume-slope", False, "hybrid", 32),
        (4, None, False, "volume-slope", False, "hybrid", 32),
        (5, None, False, "volume-slope", False, "hybrid", 32),
        (6, None, False, "volume-slope", False, "hybrid", 32),
        (7, None, False, "volume-slope", False, "hybrid", 32),
        (8, None, False, "volume-slope", False, "hybrid", 32),
        (9, None, False, "volume-slope", False, "hybrid", 32),
        (0, "linear", True, "volume-slope", True, "forward", 32),
        (1, "linear", True, "volume-slope", True, "forward", 32),
        (2, "linear", True, "volume-slope", True, "forward", 32),
        (3, "linear", True, "volume-slope", True, "forward", 32),
        (4, "linear", True, "volume-slope", True, "forward", 32),
        (5, "linear", True, "volume-slope", True, "forward", 32),
        (6, "linear", True, "volume-slope", True, "forward", 32),
        (7, "linear", True, "volume-slope", True, "forward", 32),
        (8, "linear", True, "volume-slope", True, "forward", 32),
        (9, "linear", True, "volume-slope", True, "forward", 32),
        (0, None, True, "volume-slope", True, "forward", 32),
        (1, None, True, "volume-slope", True, "forward", 32),
        (2, None, True, "volume-slope", True, "forward", 32),
        (3, None, True, "volume-slope", True, "forward", 32),
        (4, None, True, "volume-slope", True, "forward", 32),
        (5, None, True, "volume-slope", True, "forward", 32),
        (6, None, True, "volume-slope", True, "forward", 32),
        (7, None, True, "volume-slope", True, "forward", 32),
        (8, None, True, "volume-slope", True, "forward", 32),
        (9, None, True, "volume-slope", True, "forward", 32),
        (0, "linear", False, "volume-slope", True, "forward", 32),
        (1, "linear", False, "volume-slope", True, "forward", 32),
        (2, "linear", False, "volume-slope", True, "forward", 32),
        (3, "linear", False, "volume-slope", True, "forward", 32),
        (4, "linear", False, "volume-slope", True, "forward", 32),
        (5, "linear", False, "volume-slope", True, "forward", 32),
        (6, "linear", False, "volume-slope", True, "forward", 32),
        (7, "linear", False, "volume-slope", True, "forward", 32),
        (8, "linear", False, "volume-slope", True, "forward", 32),
        (9, "linear", False, "volume-slope", True, "forward", 32),
        (0, None, False, "volume-slope", True, "forward", 32),
        (1, None, False, "volume-slope", True, "forward", 32),
        (2, None, False, "volume-slope", True, "forward", 32),
        (3, None, False, "volume-slope", True, "forward", 32),
        (4, None, False, "volume-slope", True, "forward", 32),
        (5, None, False, "volume-slope", True, "forward", 32),
        (6, None, False, "volume-slope", True, "forward", 32),
        (7, None, False, "volume-slope", True, "forward", 32),
        (8, None, False, "volume-slope", True, "forward", 32),
        (9, None, False, "volume-slope", True, "forward", 32),
        (0, "relu", True, "volume-slope", False, "forward", 32),
        (1, "relu", True, "volume-slope", False, "forward", 32),
        (2, "relu", True, "volume-slope", False, "forward", 32),
        (3, "relu", True, "volume-slope", False, "forward", 32),
        (4, "relu", True, "volume-slope", False, "forward", 32),
        (5, "relu", True, "volume-slope", False, "forward", 32),
        (6, "relu", True, "volume-slope", False, "forward", 32),
        (7, "relu", True, "volume-slope", False, "forward", 32),
        (8, "relu", True, "volume-slope", False, "forward", 32),
        (9, "relu", True, "volume-slope", False, "forward", 32),
        (0, "linear", True, "volume-slope", False, "forward", 32),
        (1, "linear", True, "volume-slope", False, "forward", 32),
        (2, "linear", True, "volume-slope", False, "forward", 32),
        (3, "linear", True, "volume-slope", False, "forward", 32),
        (4, "linear", True, "volume-slope", False, "forward", 32),
        (5, "linear", True, "volume-slope", False, "forward", 32),
        (6, "linear", True, "volume-slope", False, "forward", 32),
        (7, "linear", True, "volume-slope", False, "forward", 32),
        (8, "linear", True, "volume-slope", False, "forward", 32),
        (9, "linear", True, "volume-slope", False, "forward", 32),
        (0, None, True, "volume-slope", False, "forward", 32),
        (1, None, True, "volume-slope", False, "forward", 32),
        (2, None, True, "volume-slope", False, "forward", 32),
        (3, None, True, "volume-slope", False, "forward", 32),
        (4, None, True, "volume-slope", False, "forward", 32),
        (5, None, True, "volume-slope", False, "forward", 32),
        (6, None, True, "volume-slope", False, "forward", 32),
        (7, None, True, "volume-slope", False, "forward", 32),
        (8, None, True, "volume-slope", False, "forward", 32),
        (9, None, True, "volume-slope", False, "forward", 32),
        (0, "relu", False, "volume-slope", False, "forward", 32),
        (1, "relu", False, "volume-slope", False, "forward", 32),
        (2, "relu", False, "volume-slope", False, "forward", 32),
        (3, "relu", False, "volume-slope", False, "forward", 32),
        (4, "relu", False, "volume-slope", False, "forward", 32),
        (5, "relu", False, "volume-slope", False, "forward", 32),
        (6, "relu", False, "volume-slope", False, "forward", 32),
        (7, "relu", False, "volume-slope", False, "forward", 32),
        (8, "relu", False, "volume-slope", False, "forward", 32),
        (9, "relu", False, "volume-slope", False, "forward", 32),
        (0, "linear", False, "volume-slope", False, "forward", 32),
        (1, "linear", False, "volume-slope", False, "forward", 32),
        (2, "linear", False, "volume-slope", False, "forward", 32),
        (3, "linear", False, "volume-slope", False, "forward", 32),
        (4, "linear", False, "volume-slope", False, "forward", 32),
        (5, "linear", False, "volume-slope", False, "forward", 32),
        (6, "linear", False, "volume-slope", False, "forward", 32),
        (7, "linear", False, "volume-slope", False, "forward", 32),
        (8, "linear", False, "volume-slope", False, "forward", 32),
        (9, "linear", False, "volume-slope", False, "forward", 32),
        (0, None, False, "volume-slope", False, "forward", 32),
        (1, None, False, "volume-slope", False, "forward", 32),
        (2, None, False, "volume-slope", False, "forward", 32),
        (3, None, False, "volume-slope", False, "forward", 32),
        (4, None, False, "volume-slope", False, "forward", 32),
        (5, None, False, "volume-slope", False, "forward", 32),
        (6, None, False, "volume-slope", False, "forward", 32),
        (7, None, False, "volume-slope", False, "forward", 32),
        (8, None, False, "volume-slope", False, "forward", 32),
        (9, None, False, "volume-slope", False, "forward", 32),
        (0, "linear", True, "volume-slope", True, "ibp", 32),
        (1, "linear", True, "volume-slope", True, "ibp", 32),
        (2, "linear", True, "volume-slope", True, "ibp", 32),
        (3, "linear", True, "volume-slope", True, "ibp", 32),
        (4, "linear", True, "volume-slope", True, "ibp", 32),
        (5, "linear", True, "volume-slope", True, "ibp", 32),
        (6, "linear", True, "volume-slope", True, "ibp", 32),
        (7, "linear", True, "volume-slope", True, "ibp", 32),
        (8, "linear", True, "volume-slope", True, "ibp", 32),
        (9, "linear", True, "volume-slope", True, "ibp", 32),
        (0, None, True, "volume-slope", True, "ibp", 32),
        (1, None, True, "volume-slope", True, "ibp", 32),
        (2, None, True, "volume-slope", True, "ibp", 32),
        (3, None, True, "volume-slope", True, "ibp", 32),
        (4, None, True, "volume-slope", True, "ibp", 32),
        (5, None, True, "volume-slope", True, "ibp", 32),
        (6, None, True, "volume-slope", True, "ibp", 32),
        (7, None, True, "volume-slope", True, "ibp", 32),
        (8, None, True, "volume-slope", True, "ibp", 32),
        (9, None, True, "volume-slope", True, "ibp", 32),
        (0, "relu", False, "volume-slope", True, "ibp", 32),
        (1, "relu", False, "volume-slope", True, "ibp", 32),
        (2, "relu", False, "volume-slope", True, "ibp", 32),
        (3, "relu", False, "volume-slope", True, "ibp", 32),
        (4, "relu", False, "volume-slope", True, "ibp", 32),
        (5, "relu", False, "volume-slope", True, "ibp", 32),
        (6, "relu", False, "volume-slope", True, "ibp", 32),
        (7, "relu", False, "volume-slope", True, "ibp", 32),
        (8, "relu", False, "volume-slope", True, "ibp", 32),
        (9, "relu", False, "volume-slope", True, "ibp", 32),
        (0, "linear", False, "volume-slope", True, "ibp", 32),
        (1, "linear", False, "volume-slope", True, "ibp", 32),
        (2, "linear", False, "volume-slope", True, "ibp", 32),
        (3, "linear", False, "volume-slope", True, "ibp", 32),
        (4, "linear", False, "volume-slope", True, "ibp", 32),
        (5, "linear", False, "volume-slope", True, "ibp", 32),
        (6, "linear", False, "volume-slope", True, "ibp", 32),
        (7, "linear", False, "volume-slope", True, "ibp", 32),
        (8, "linear", False, "volume-slope", True, "ibp", 32),
        (9, "linear", False, "volume-slope", True, "ibp", 32),
        (0, None, False, "volume-slope", True, "ibp", 32),
        (1, None, False, "volume-slope", True, "ibp", 32),
        (2, None, False, "volume-slope", True, "ibp", 32),
        (3, None, False, "volume-slope", True, "ibp", 32),
        (4, None, False, "volume-slope", True, "ibp", 32),
        (5, None, False, "volume-slope", True, "ibp", 32),
        (6, None, False, "volume-slope", True, "ibp", 32),
        (7, None, False, "volume-slope", True, "ibp", 32),
        (8, None, False, "volume-slope", True, "ibp", 32),
        (9, None, False, "volume-slope", True, "ibp", 32),
        (0, "linear", True, "volume-slope", False, "ibp", 32),
        (1, "linear", True, "volume-slope", False, "ibp", 32),
        (2, "linear", True, "volume-slope", False, "ibp", 32),
        (3, "linear", True, "volume-slope", False, "ibp", 32),
        (4, "linear", True, "volume-slope", False, "ibp", 32),
        (5, "linear", True, "volume-slope", False, "ibp", 32),
        (6, "linear", True, "volume-slope", False, "ibp", 32),
        (7, "linear", True, "volume-slope", False, "ibp", 32),
        (8, "linear", True, "volume-slope", False, "ibp", 32),
        (9, "linear", True, "volume-slope", False, "ibp", 32),
        (0, None, True, "volume-slope", False, "ibp", 32),
        (1, None, True, "volume-slope", False, "ibp", 32),
        (2, None, True, "volume-slope", False, "ibp", 32),
        (3, None, True, "volume-slope", False, "ibp", 32),
        (4, None, True, "volume-slope", False, "ibp", 32),
        (5, None, True, "volume-slope", False, "ibp", 32),
        (6, None, True, "volume-slope", False, "ibp", 32),
        (7, None, True, "volume-slope", False, "ibp", 32),
        (8, None, True, "volume-slope", False, "ibp", 32),
        (9, None, True, "volume-slope", False, "ibp", 32),
        (0, "linear", False, "volume-slope", False, "ibp", 32),
        (1, "linear", False, "volume-slope", False, "ibp", 32),
        (2, "linear", False, "volume-slope", False, "ibp", 32),
        (3, "linear", False, "volume-slope", False, "ibp", 32),
        (4, "linear", False, "volume-slope", False, "ibp", 32),
        (5, "linear", False, "volume-slope", False, "ibp", 32),
        (6, "linear", False, "volume-slope", False, "ibp", 32),
        (7, "linear", False, "volume-slope", False, "ibp", 32),
        (8, "linear", False, "volume-slope", False, "ibp", 32),
        (9, "linear", False, "volume-slope", False, "ibp", 32),
        (0, None, False, "volume-slope", False, "ibp", 32),
        (1, None, False, "volume-slope", False, "ibp", 32),
        (2, None, False, "volume-slope", False, "ibp", 32),
        (3, None, False, "volume-slope", False, "ibp", 32),
        (4, None, False, "volume-slope", False, "ibp", 32),
        (5, None, False, "volume-slope", False, "ibp", 32),
        (6, None, False, "volume-slope", False, "ibp", 32),
        (7, None, False, "volume-slope", False, "ibp", 32),
        (8, None, False, "volume-slope", False, "ibp", 32),
        (9, None, False, "volume-slope", False, "ibp", 32),
        (1, None, True, "volume-slope", True, "hybrid", 16),
        (2, None, True, "volume-slope", True, "hybrid", 16),
        (3, None, True, "volume-slope", True, "hybrid", 16),
        (4, None, True, "volume-slope", True, "hybrid", 16),
        (5, None, True, "volume-slope", True, "hybrid", 16),
        (6, None, True, "volume-slope", True, "hybrid", 16),
        (7, None, True, "volume-slope", True, "hybrid", 16),
        (8, None, True, "volume-slope", True, "hybrid", 16),
        (9, None, True, "volume-slope", True, "hybrid", 16),
        (0, "relu", False, "volume-slope", True, "hybrid", 16),
        (1, "relu", False, "volume-slope", True, "hybrid", 16),
        (2, "relu", False, "volume-slope", True, "hybrid", 16),
        (3, "relu", False, "volume-slope", True, "hybrid", 16),
        (4, "relu", False, "volume-slope", True, "hybrid", 16),
    ],
)
def test_Backward_Dense_1D_box(n, activation, use_bias, slope, previous, mode, floatx):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    layer_ = Dense(1, use_bias=use_bias, activation=activation, dtype=K.floatx())
    input_dim = x.shape[-1]
    layer_(inputs[1])
    if mode == "hybrid":
        IBP = True
        forward = True
    if mode == "ibp":
        IBP = True
        forward = False
    if mode == "forward":
        IBP = False
        forward = True

    layer = to_monotonic(layer_, (2, input_dim), dc_decomp=False, IBP=IBP, forward=forward, shared=True)[0]

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

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1,), dtype=K.floatx())
    # get backward layer
    layer_backward = get_backward(
        layer_, input_dim=input_dim, slope=slope, previous=previous, mode=mode, convex_domain={}
    )

    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode + [w_out, b_out, w_out, b_out])

        f_dense = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])

        output_ = f_dense(inputs_ + [np.ones((len(x), 1, 1)), np.zeros((len(x), 1))])
    else:

        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_dense(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_

    if use_bias:
        W_, bias = layer.get_weights()
    else:
        W_ = layer.get_weights()[0]
        bias = 0.0 * W_[0]

    assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )
    if use_bias:
        layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    else:
        layer.set_weights([2 * np.ones_like(W_)])

    assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )
    if use_bias:
        layer.set_weights([-2 * np.ones_like(W_), np.ones_like(bias)])
    else:
        layer.set_weights([-2 * np.ones_like(W_)])

    assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "odd, activation, floatx, mode, previous",
    [
        (0, None, 32, "hybrid", True),
        (1, None, 32, "hybrid", True),
        (0, "linear", 32, "hybrid", True),
        (1, "linear", 32, "hybrid", True),
        (0, "relu", 32, "hybrid", True),
        (1, "relu", 32, "hybrid", True),
        (0, None, 64, "hybrid", True),
        (1, None, 64, "hybrid", True),
        (0, "linear", 64, "hybrid", True),
        (1, "linear", 64, "hybrid", True),
        (0, "relu", 64, "hybrid", True),
        (1, "relu", 64, "hybrid", True),
        (0, None, 16, "hybrid", True),
        (1, None, 16, "hybrid", True),
        (0, "linear", 16, "hybrid", True),
        (1, "linear", 16, "hybrid", True),
        (0, "relu", 16, "hybrid", True),
        (1, "relu", 16, "hybrid", True),
        (0, None, 32, "forward", True),
        (1, None, 32, "forward", True),
        (0, "linear", 32, "forward", True),
        (1, "linear", 32, "forward", True),
        (0, "relu", 32, "forward", True),
        (1, "relu", 32, "forward", True),
        (0, None, 64, "forward", True),
        (1, None, 64, "forward", True),
        (0, "linear", 64, "forward", True),
        (1, "linear", 64, "forward", True),
        (0, "relu", 64, "forward", True),
        (1, "relu", 64, "forward", True),
        (0, None, 16, "forward", True),
        (1, None, 16, "forward", True),
        (0, "linear", 16, "forward", True),
        (1, "linear", 16, "forward", True),
        (0, "relu", 16, "forward", True),
        (1, "relu", 16, "forward", True),
        (0, None, 32, "ibp", True),
        (1, None, 32, "ibp", True),
        (0, "linear", 32, "ibp", True),
        (1, "linear", 32, "ibp", True),
        (0, "relu", 32, "ibp", True),
        (1, "relu", 32, "ibp", True),
        (0, None, 64, "ibp", True),
        (1, None, 64, "ibp", True),
        (0, "linear", 64, "ibp", True),
        (1, "linear", 64, "ibp", True),
        (0, "relu", 64, "ibp", True),
        (1, "relu", 64, "ibp", True),
        (0, None, 16, "ibp", True),
        (1, None, 16, "ibp", True),
        (0, "linear", 16, "ibp", True),
        (1, "linear", 16, "ibp", True),
        (0, "relu", 16, "ibp", True),
        (1, "relu", 16, "ibp", True),
        (0, None, 32, "hybrid", False),
        (1, None, 32, "hybrid", False),
        (0, "linear", 32, "hybrid", False),
        (1, "linear", 32, "hybrid", False),
        (0, "relu", 32, "hybrid", False),
        (1, "relu", 32, "hybrid", False),
        (0, None, 64, "hybrid", False),
        (1, None, 64, "hybrid", False),
        (0, "linear", 64, "hybrid", False),
        (1, "linear", 64, "hybrid", False),
        (0, "relu", 64, "hybrid", False),
        (1, "relu", 64, "hybrid", False),
        (0, None, 16, "hybrid", False),
        (1, None, 16, "hybrid", False),
        (0, "linear", 16, "hybrid", False),
        (1, "linear", 16, "hybrid", False),
        (0, "relu", 16, "hybrid", False),
        (1, "relu", 16, "hybrid", False),
        (0, None, 32, "forward", False),
        (1, None, 32, "forward", False),
        (0, "linear", 32, "forward", False),
        (1, "linear", 32, "forward", False),
        (0, "relu", 32, "forward", False),
        (1, "relu", 32, "forward", False),
        (0, None, 64, "forward", False),
        (1, None, 64, "forward", False),
        (0, "linear", 64, "forward", False),
        (1, "linear", 64, "forward", False),
        (0, "relu", 64, "forward", False),
        (1, "relu", 64, "forward", False),
        (0, None, 16, "forward", False),
        (1, None, 16, "forward", False),
        (0, "linear", 16, "forward", False),
        (1, "linear", 16, "forward", False),
        (0, "relu", 16, "forward", False),
        (1, "relu", 16, "forward", False),
        (0, None, 32, "ibp", False),
        (1, None, 32, "ibp", False),
        (0, "linear", 32, "ibp", False),
        (1, "linear", 32, "ibp", False),
        (0, "relu", 32, "ibp", False),
        (1, "relu", 32, "ibp", False),
        (0, None, 64, "ibp", False),
        (1, None, 64, "ibp", False),
        (0, "linear", 64, "ibp", False),
        (1, "linear", 64, "ibp", False),
        (0, "relu", 64, "ibp", False),
        (1, "relu", 64, "ibp", False),
        (0, None, 16, "ibp", False),
        (1, None, 16, "ibp", False),
        (0, "linear", 16, "ibp", False),
        (1, "linear", 16, "ibp", False),
        (0, "relu", 16, "ibp", False),
        (1, "relu", 16, "ibp", False),
    ],
)
def test_Backward_DecomonDense_multiD_box(odd, activation, floatx, mode, previous):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    input_dim = x.shape[-1]
    layer_ = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())
    layer_(inputs[1])
    if mode == "hybrid":
        IBP = True
        forward = True
    if mode == "ibp":
        IBP = True
        forward = False
    if mode == "forward":
        IBP = False
        forward = True

    layer = to_monotonic(layer_, (2, input_dim), dc_decomp=False, IBP=IBP, forward=forward, shared=True)[0]

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

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1,), dtype=K.floatx())
    # get backward layer
    layer_backward = get_backward(layer_, input_dim=input_dim, previous=previous, mode=mode, convex_domain={})
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
    "n, activation",
    [(0, "relu")],
)
def test_Backward_DecomonDense_1D_box_model(n, activation):
    layer = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[2:])
    z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1))
    b_out = Input((1,))
    # get backward layer
    layer_backward = get_backward(layer, input_dim=1)
    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = layer_backward(inputs[2:] + [w_out, b_out, w_out, b_out])

    Model(inputs[2:] + [w_out, b_out], [w_out_u_, b_out_u_, w_out_l_, b_out_l_])
