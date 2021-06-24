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
from tensorflow.keras.models import Model


@pytest.mark.parametrize(
    "n, activation, n_subgrad, slope",
    [
        (0, "relu", 0, "volume-slope"),
        (1, "relu", 0, "volume-slope"),
        (2, "relu", 0, "volume-slope"),
        (3, "relu", 0, "volume-slope"),
        (4, "relu", 0, "volume-slope"),
        (5, "relu", 0, "volume-slope"),
        (6, "relu", 0, "volume-slope"),
        (7, "relu", 0, "volume-slope"),
        (8, "relu", 0, "volume-slope"),
        (9, "relu", 0, "volume-slope"),
        (0, "linear", 0, "volume-slope"),
        (1, "linear", 0, "volume-slope"),
        (2, "linear", 0, "volume-slope"),
        (3, "linear", 0, "volume-slope"),
        (4, "linear", 0, "volume-slope"),
        (5, "linear", 0, "volume-slope"),
        (6, "linear", 0, "volume-slope"),
        (7, "linear", 0, "volume-slope"),
        (8, "linear", 0, "volume-slope"),
        (9, "linear", 0, "volume-slope"),
        (0, None, 0, "volume-slope"),
        (1, None, 0, "volume-slope"),
        (2, None, 0, "volume-slope"),
        (3, None, 0, "volume-slope"),
        (4, None, 0, "volume-slope"),
        (5, None, 0, "volume-slope"),
        (6, None, 0, "volume-slope"),
        (7, None, 0, "volume-slope"),
        (8, None, 0, "volume-slope"),
        (9, None, 0, "volume-slope"),
        (0, "relu", 1, "volume-slope"),
        (1, "relu", 1, "volume-slope"),
        (2, "relu", 1, "volume-slope"),
        (3, "relu", 1, "volume-slope"),
        (4, "relu", 1, "volume-slope"),
        (5, "relu", 1, "volume-slope"),
        (6, "relu", 1, "volume-slope"),
        (7, "relu", 1, "volume-slope"),
        (8, "relu", 1, "volume-slope"),
        (9, "relu", 1, "volume-slope"),
        (0, "linear", 1, "volume-slope"),
        (1, "linear", 1, "volume-slope"),
        (2, "linear", 1, "volume-slope"),
        (3, "linear", 1, "volume-slope"),
        (4, "linear", 1, "volume-slope"),
        (5, "linear", 1, "volume-slope"),
        (6, "linear", 1, "volume-slope"),
        (7, "linear", 1, "volume-slope"),
        (8, "linear", 1, "volume-slope"),
        (9, "linear", 1, "volume-slope"),
        (0, None, 1, "volume-slope"),
        (1, None, 1, "volume-slope"),
        (2, None, 1, "volume-slope"),
        (3, None, 1, "volume-slope"),
        (4, None, 1, "volume-slope"),
        (5, None, 1, "volume-slope"),
        (6, None, 1, "volume-slope"),
        (7, None, 1, "volume-slope"),
        (8, None, 1, "volume-slope"),
        (9, None, 1, "volume-slope"),
        (0, "relu", 5, "volume-slope"),
        (1, "relu", 5, "volume-slope"),
        (2, "relu", 5, "volume-slope"),
        (3, "relu", 5, "volume-slope"),
        (4, "relu", 5, "volume-slope"),
        (5, "relu", 5, "volume-slope"),
        (6, "relu", 5, "volume-slope"),
        (7, "relu", 5, "volume-slope"),
        (8, "relu", 5, "volume-slope"),
        (9, "relu", 5, "volume-slope"),
        (0, "linear", 5, "volume-slope"),
        (1, "linear", 5, "volume-slope"),
        (2, "linear", 5, "volume-slope"),
        (3, "linear", 5, "volume-slope"),
        (4, "linear", 5, "volume-slope"),
        (5, "linear", 5, "volume-slope"),
        (6, "linear", 5, "volume-slope"),
        (7, "linear", 5, "volume-slope"),
        (8, "linear", 5, "volume-slope"),
        (9, "linear", 5, "volume-slope"),
        (0, None, 5, "volume-slope"),
        (1, None, 5, "volume-slope"),
        (2, None, 5, "volume-slope"),
        (3, None, 5, "volume-slope"),
        (4, None, 5, "volume-slope"),
        (5, None, 5, "volume-slope"),
        (6, None, 5, "volume-slope"),
        (7, None, 5, "volume-slope"),
        (8, None, 5, "volume-slope"),
        (9, None, 5, "volume-slope"),
        (0, "relu", 1, "same-slope"),
        (1, "relu", 1, "same-slope"),
        (2, "relu", 1, "same-slope"),
        (3, "relu", 1, "same-slope"),
        (4, "relu", 1, "same-slope"),
        (5, "relu", 1, "same-slope"),
        (6, "relu", 1, "same-slope"),
        (7, "relu", 1, "same-slope"),
        (8, "relu", 1, "same-slope"),
        (9, "relu", 1, "same-slope"),
        (0, "relu", 1, "zero-lb"),
        (1, "relu", 1, "zero-lb"),
        (2, "relu", 1, "zero-lb"),
        (3, "relu", 1, "zero-lb"),
        (4, "relu", 1, "zero-lb"),
        (5, "relu", 1, "zero-lb"),
        (6, "relu", 1, "zero-lb"),
        (7, "relu", 1, "zero-lb"),
        (8, "relu", 1, "zero-lb"),
        (9, "relu", 1, "zero-lb"),
        (0, "relu", 1, "one-lb"),
        (1, "relu", 1, "one-lb"),
        (2, "relu", 1, "one-lb"),
        (3, "relu", 1, "one-lb"),
        (4, "relu", 1, "one-lb"),
        (5, "relu", 1, "one-lb"),
        (6, "relu", 1, "one-lb"),
        (7, "relu", 1, "one-lb"),
        (8, "relu", 1, "one-lb"),
        (9, "relu", 1, "one-lb"),
    ],
)
def test_Backward_DecomonDense_1D_box(n, activation, n_subgrad, slope):
    layer = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False, n_subgrad=n_subgrad)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[1:])
    y_0, z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))
    # get backward layer
    layer_backward = get_backward(layer, slope=slope)
    w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(inputs[1:] + [w_out, b_out, w_out, b_out])

    f_dense = K.function(inputs + [w_out, b_out], [y_0, z_0, u_c_0, w_out_u, b_out_u, l_c_0, w_out_l, b_out_l])

    output_ = f_dense(inputs_ + [np.ones((len(x), 1, 1, 1)), np.zeros((len(x), 1, 1))])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_

    W_, bias = layer.get_weights()

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

    layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

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

    layer.set_weights([-2 * np.ones_like(W_), np.ones_like(bias)])

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


@pytest.mark.parametrize(
    "odd, activation, n_subgrad",
    [
        (0, None, 0),
        (1, None, 0),
        (0, "linear", 0),
        (1, "linear", 0),
        (0, "relu", 0),
        (1, "relu", 0),
        (0, None, 1),
        (1, None, 1),
        (0, "linear", 1),
        (1, "linear", 1),
        (0, "relu", 1),
        (1, "relu", 1),
        (0, None, 5),
        (1, None, 5),
        (0, "linear", 5),
        (1, "linear", 5),
        (0, "relu", 5),
        (1, "relu", 5),
    ],
)
def test_Backward_DecomonDense_multiD_box(odd, activation, n_subgrad):
    layer = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False, n_subgrad=n_subgrad)
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

    W_, bias = layer.get_weights()

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


@pytest.mark.parametrize(
    "n, activation, n_subgrad",
    [
        (0, "relu", 0),
    ],
)
def test_Backward_DecomonDense_1D_box_model(n, activation, n_subgrad):
    layer = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False, n_subgrad=n_subgrad)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[1:])
    y_0, z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))
    # get backward layer
    layer_backward = get_backward(layer)
    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = layer_backward(inputs[1:] + [w_out, b_out, w_out, b_out])

    Model(inputs[1:] + [w_out, b_out], [w_out_u_, b_out_u_, w_out_l_, b_out_l_])
