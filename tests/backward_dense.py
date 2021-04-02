# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
import tensorflow.python.keras.backend as K
from decomon.backward_layers.backward_layers import get_backward
from decomon.layers.decomon_layers import DecomonDense
from tensorflow.keras.layers import Input
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_tensor_decomposition_multid_box,
)


@pytest.mark.parametrize(
    "n, activation",
    [
        (0, "relu"),
        (1, "relu"),
        (2, "relu"),
        (3, "relu"),
        (4, "relu"),
        (5, "relu"),
        (6, "relu"),
        (7, "relu"),
        (8, "relu"),
        (9, "relu"),
        (0, "linear"),
        (1, "linear"),
        (2, "linear"),
        (3, "linear"),
        (4, "linear"),
        (5, "linear"),
        (6, "linear"),
        (7, "linear"),
        (8, "linear"),
        (9, "linear"),
        (0, None),
        (1, None),
        (2, None),
        (3, None),
        (4, None),
        (5, None),
        (6, None),
        (7, None),
        (8, None),
        (9, None),
    ],
)
def test_Backward_DecomonDense_1D_box(n, activation):
    layer = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[1:])
    y_0, z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))
    # get backward layer
    layer_backward = get_backward(layer)
    w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(inputs[1:] + [w_out, b_out, w_out, b_out])

    f_dense = K.function(inputs + [w_out, b_out], [y_0, z_0, u_c_0, w_out_u, b_out_u, l_c_0, w_out_l, b_out_l])

    output_ = f_dense(inputs_ + [np.ones((len(x), 1, 1, 1)), np.zeros((len(x), 1, 1))])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_

    W_pos, W_neg, bias = layer.get_weights()

    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        l_c_,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )

    layer.set_weights([2 * np.ones_like(W_pos), np.zeros_like(W_neg), np.ones_like(bias)])

    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        l_c_,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )

    layer.set_weights([np.zeros_like(W_neg), -2 * np.ones_like(W_pos), np.ones_like(bias)])

    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        l_c_,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )


# multi dimensional
@pytest.mark.parametrize(
    "odd, activation", [(0, None), (1, None), (0, "linear"), (1, "linear"), (0, "relu"), (1, "relu")]
)
def test_Backward_DecomonDense_multiD_box(odd, activation):
    layer = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[1:])
    y_0, z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))
    # get backward layer
    layer_backward = get_backward(layer)
    w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(inputs[1:] + [w_out, b_out, w_out, b_out])

    f_dense = K.function(inputs + [w_out, b_out], [y_0, z_0, u_c_0, w_out_u, b_out_u, l_c_0, w_out_l, b_out_l])

    output_ = f_dense(inputs_ + [np.ones((len(x), 1, 1, 1)), np.zeros((len(x), 1, 1))])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_

    W_pos, W_neg, bias = layer.get_weights()

    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        l_c_,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(odd),
    )
