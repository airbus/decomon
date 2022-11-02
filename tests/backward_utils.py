# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from tensorflow.keras.layers import Input
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    get_tensor_decomposition_multid_box,
    get_standard_values_multid_box,
    assert_output_properties_box_linear,
)
import tensorflow.python.keras.backend as K

from decomon.layers.utils import relu_, max_, maximum, add, minus, substract, softplus_
from decomon.backward_layers.utils import (
    backward_relu_,
    backward_max_,
    backward_add,
    backward_minus,
    backward_maximum,
    backward_substract,
    backward_softplus_,
)


@pytest.mark.parametrize(
    "n, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
    ],
)
def test_relu_backward_1D_box(n, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs

    x_, y_, z_, u_c_, W_u_, B_u_, l_c_, W_l_, B_l_ = inputs_

    if mode == "hybrid":
        input_mode = inputs[2:]
        output = relu_(input_mode, dc_decomp=False, mode=mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode == "forward":
        input_mode = [z, W_u, b_u, W_l, b_l]
        output = relu_(input_mode, dc_decomp=False, mode=mode)
        z_0, _, _, _, _ = output
    if mode == "ibp":
        input_mode = [u_c, l_c]
        output = relu_(input_mode, dc_decomp=False, mode=mode)
        z_0 = z_
        u_c_0, l_c_0 = output

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1), dtype=K.floatx())

    # backward_relu_(input_mode, w_out, b_out, w_out, b_out, mode=mode)
    w_out_u, b_out_u, w_out_l, b_out_l = backward_relu_(input_mode, w_out, b_out, w_out, b_out, mode=mode)

    f_relu = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
    output_ = f_relu(inputs_ + [np.ones((len(x_), 1, 1)), np.zeros((len(x_), 1))])
    w_u_, b_u_, w_l_, b_l_ = output_

    w_u_b = np.sum(np.maximum(w_u_, 0) * W_u_ + np.minimum(w_u_, 0) * W_l_, 1)[:, :, None]
    b_u_b = b_u_ + np.sum(np.maximum(w_u_, 0) * B_u_[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * B_l_[:, :, None], 1)
    w_l_b = np.sum(np.maximum(w_l_, 0) * W_l_ + np.minimum(w_l_, 0) * W_u_, 1)[:, :, None]
    b_l_b = (
        b_l_
        + np.sum(np.maximum(w_l_[:, 0], 0) * B_l_[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * B_u_[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_, None, z_[:, 0], z_[:, 1], None, w_u_b, b_u_b, None, w_l_b, b_l_b, "dense_{}".format(n), decimal=decimal
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")



@pytest.mark.parametrize(
    "n_0, n_1, mode, floatx",
    [
        (0, 3, "hybrid", 32),
        (1, 4, "hybrid", 32),
        (2, 5, "hybrid", 32),
        (0, 3, "hybrid", 64),
        (1, 4, "hybrid", 64),
        (2, 5, "hybrid", 64),
        (0, 3, "hybrid", 16),
        (1, 4, "hybrid", 16),
        (2, 5, "hybrid", 16),
        (0, 3, "forward", 32),
        (1, 4, "forward", 32),
        (2, 5, "forward", 32),
        (0, 3, "forward", 64),
        (1, 4, "forward", 64),
        (2, 5, "forward", 64),
        (0, 3, "forward", 16),
        (1, 4, "forward", 16),
        (2, 5, "forward", 16),
        (0, 3, "ibp", 32),
        (1, 4, "ibp", 32),
        (2, 5, "ibp", 32),
        (0, 3, "ibp", 64),
        (1, 4, "ibp", 64),
        (2, 5, "ibp", 64),
        (0, 3, "ibp", 16),
        (1, 4, "ibp", 16),
        (2, 5, "ibp", 16),
    ],
)
def test_add_backward_1D_box(n_0, n_1, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1,), dtype=K.floatx())

    if mode == "hybrid":
        input_tmp_0 = inputs_0[2:]
        input_tmp_1 = inputs_1[2:]
    if mode == "ibp":
        input_tmp_0 = [inputs_0[3], inputs_0[6]]
        input_tmp_1 = [inputs_1[3], inputs_1[6]]
    if mode == "forward":
        input_tmp_0 = [inputs_0[2], inputs_0[4], inputs_0[5], inputs_0[7], inputs_0[8]]
        input_tmp_1 = [inputs_1[2], inputs_1[4], inputs_1[5], inputs_1[7], inputs_1[8]]

    back_bounds_0, back_bounds_1 = backward_add(input_tmp_0, input_tmp_1, w_out, b_out, w_out, b_out, mode=mode)
    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1)), np.zeros((len(x_0), 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_, 0) * W_u_0 + np.minimum(w_u_0_, 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_
        + np.sum(np.maximum(w_u_0_, 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_, 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_, 0) * W_l_0 + np.minimum(w_l_0_, 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_
        + np.sum(np.maximum(w_l_0_, 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_, 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_, 0) * W_u_1 + np.minimum(w_u_1_, 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_
        + np.sum(np.maximum(w_u_1_, 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_, 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_, 0) * W_l_1 + np.minimum(w_l_1_, 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_
        + np.sum(np.maximum(w_l_1_, 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_, 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        y_0 + y_1,
        z_0[:, 0],
        z_0[:, 1],
        u_c_0 + u_c_1,
        w_u_b_0,
        b_u_b_0,
        l_c_0 + l_c_1,
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
        decimal=decimal,
    )

    assert_output_properties_box_linear(
        x_1,
        y_0 + y_1,
        z_1[:, 0],
        z_1[:, 1],
        u_c_0 + u_c_1,
        w_u_b_1,
        b_u_b_1,
        l_c_0 + l_c_1,
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
        decimal=decimal,
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "n_0, n_1, mode, floatx",
    [
        (0, 3, "hybrid", 32),
        (1, 4, "hybrid", 32),
        (2, 5, "hybrid", 32),
        (0, 3, "hybrid", 64),
        (1, 4, "hybrid", 64),
        (2, 5, "hybrid", 64),
        (0, 3, "hybrid", 16),
        (1, 4, "hybrid", 16),
        (2, 5, "hybrid", 16),
        (0, 3, "forward", 32),
        (1, 4, "forward", 32),
        (2, 5, "forward", 32),
        (0, 3, "forward", 64),
        (1, 4, "forward", 64),
        (2, 5, "forward", 64),
        (0, 3, "forward", 16),
        (1, 4, "forward", 16),
        (2, 5, "forward", 16),
        (0, 3, "ibp", 32),
        (1, 4, "ibp", 32),
        (2, 5, "ibp", 32),
        (0, 3, "ibp", 64),
        (1, 4, "ibp", 64),
        (2, 5, "ibp", 64),
        (0, 3, "ibp", 16),
        (1, 4, "ibp", 16),
        (2, 5, "ibp", 16),
    ],
)
def test_substract_backward_1D_box(n_0, n_1, mode, floatx):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    if mode == "hybrid":
        input_tmp_0 = inputs_0[2:]
        input_tmp_1 = inputs_1[2:]
    if mode == "ibp":
        input_tmp_0 = [inputs_0[3], inputs_0[6]]
        input_tmp_1 = [inputs_1[3], inputs_1[6]]
    if mode == "forward":
        input_tmp_0 = [inputs_0[2], inputs_0[4], inputs_0[5], inputs_0[7], inputs_0[8]]
        input_tmp_1 = [inputs_1[2], inputs_1[4], inputs_1[5], inputs_1[7], inputs_1[8]]

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1,), dtype=K.floatx())

    back_bounds_0, back_bounds_1 = backward_substract(input_tmp_0, input_tmp_1, w_out, b_out, w_out, b_out, mode=mode)

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1)), np.zeros((len(x_0), 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    f_ref = K.function(inputs_0 + inputs_1, substract(inputs_0[2:], inputs_1[2:]))

    _, u_, _, _, l_, _, _ = f_ref(inputs_0_ + inputs_1_)

    w_u_b_0 = np.sum(np.maximum(w_u_0_, 0) * W_u_0 + np.minimum(w_u_0_, 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_
        + np.sum(np.maximum(w_u_0_, 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_, 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_, 0) * W_l_0 + np.minimum(w_l_0_, 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_
        + np.sum(np.maximum(w_l_0_, 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_, 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_, 0) * W_u_1 + np.minimum(w_u_1_, 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_
        + np.sum(np.maximum(w_u_1_, 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_, 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_, 0) * W_l_1 + np.minimum(w_l_1_, 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_
        + np.sum(np.maximum(w_l_1_, 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_, 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        y_0 - y_1,
        z_0[:, 0],
        z_0[:, 1],
        u_c_0 - l_c_1,
        w_u_b_0,
        b_u_b_0,
        l_c_0 - u_c_1,
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
        decimal=decimal,
    )

    assert_output_properties_box_linear(
        x_1,
        y_0 - y_1,
        z_1[:, 0],
        z_1[:, 1],
        u_c_0 - l_c_1,
        w_u_b_1,
        b_u_b_1,
        l_c_0 - u_c_1,
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
        decimal=decimal,
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "n_0, n_1, mode, floatx",
    [
        (0, 3, "hybrid", 32),
        (1, 4, "hybrid", 32),
        (2, 5, "hybrid", 32),
        (0, 3, "hybrid", 64),
        (1, 4, "hybrid", 64),
        (2, 5, "hybrid", 64),
        (0, 3, "hybrid", 16),
        (1, 4, "hybrid", 16),
        (2, 5, "hybrid", 16),
        (0, 3, "forward", 32),
        (1, 4, "forward", 32),
        (2, 5, "forward", 32),
        (0, 3, "forward", 64),
        (1, 4, "forward", 64),
        (2, 5, "forward", 64),
        (0, 3, "forward", 16),
        (1, 4, "forward", 16),
        (2, 5, "forward", 16),
        (0, 3, "ibp", 32),
        (1, 4, "ibp", 32),
        (2, 5, "ibp", 32),
        (0, 3, "ibp", 64),
        (1, 4, "ibp", 64),
        (2, 5, "ibp", 64),
        (0, 3, "ibp", 16),
        (1, 4, "ibp", 16),
        (2, 5, "ibp", 16),
    ],
)
def test_maximum_backward_1D_box(n_0, n_1, mode, floatx):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    if mode == "hybrid":
        input_tmp_0 = inputs_0[2:]
        input_tmp_1 = inputs_1[2:]
    if mode == "ibp":
        input_tmp_0 = [inputs_0[3], inputs_0[6]]
        input_tmp_1 = [inputs_1[3], inputs_1[6]]
    if mode == "forward":
        input_tmp_0 = [inputs_0[2], inputs_0[4], inputs_0[5], inputs_0[7], inputs_0[8]]
        input_tmp_1 = [inputs_1[2], inputs_1[4], inputs_1[5], inputs_1[7], inputs_1[8]]

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1,), dtype=K.floatx())

    back_bounds_0, back_bounds_1 = backward_maximum(input_tmp_0, input_tmp_1, w_out, b_out, w_out, b_out, mode=mode)

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1)), np.zeros((len(x_0), 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_, 0) * W_u_0 + np.minimum(w_u_0_, 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_
        + np.sum(np.maximum(w_u_0_, 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_, 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_, 0) * W_l_0 + np.minimum(w_l_0_, 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_
        + np.sum(np.maximum(w_l_0_, 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_, 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_, 0) * W_u_1 + np.minimum(w_u_1_, 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_
        + np.sum(np.maximum(w_u_1_, 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_, 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_, 0) * W_l_1 + np.minimum(w_l_1_, 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_
        + np.sum(np.maximum(w_l_1_, 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_, 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        np.maximum(y_0, y_1),
        z_0[:, 0],
        z_0[:, 1],
        np.maximum(u_c_0, u_c_1),
        w_u_b_0,
        b_u_b_0,
        np.maximum(l_c_0, l_c_1),
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
        decimal=decimal,
    )

    assert_output_properties_box_linear(
        x_1,
        np.maximum(y_0, y_1),
        z_1[:, 0],
        z_1[:, 1],
        np.maximum(u_c_0, u_c_1),
        w_u_b_1,
        b_u_b_1,
        np.maximum(l_c_0, l_c_1),
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
        decimal=decimal,
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


"""
@pytest.mark.parametrize("odd, mode, floatx", [(1, "hybrid", 32),
                                               #(1, "forward", 32), (1, "ibp", 32),
                                               #(1, "hybrid", 64), (1, "forward", 64), (1, "ibp", 64),
                                               #(1, "hybrid", 16), (1, "forward", 16), (1, "ibp", 16),
                                               ])
def test_max_backward_multiD_box(odd, mode, floatx):

    K.set_floatx('float{}'.format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x_, y_, z_, u_c_, W_u, b_u, l_c_, W_l, b_l = inputs_

    w_out = Input((1, 1))
    b_out = Input((1, ))

    back_bounds = backward_max_(inputs[2:], w_out, b_out, w_out, b_out)

    f_add = K.function(inputs + [w_out, b_out], back_bounds)
    output_ = f_add(inputs_ + [np.ones((len(x_), 1, 1)), np.zeros((len(x_), 1))])

    w_u_0_, b_u_0_, w_l_0_, b_l_0_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_, 0)[:,None] * W_u[:,:,:,None] + np.minimum(w_u_0_, 0)[:,None] * W_l[:,:,:,None], 2)[:, :, None]

    b_u_b_0 = (
        b_u_0_
        + np.sum(np.maximum(w_u_0_, 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_, 0) * b_l[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_, 0)[:,None] * W_l[:,:,:,None] + np.minimum(w_l_0_, 0)[:,None] * W_u[:,:,:,None], 2)[:, :, None]
    b_l_b_0 = (
        b_l_0_
        + np.sum(np.maximum(w_l_0_, 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_, 0) * b_u[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_,
        np.max(y_, -1),
        z_[:, 0],
        z_[:, 1],
        np.max(u_c_, -1),
        w_u_b_0,
        b_u_b_0,
        np.max(l_c_, -1),
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(odd),
    )

    K.set_epsilon(eps)
    K.set_floatx('float32')
"""


"""
@pytest.mark.parametrize("n, mode, floatx", [(0, "hybrid", 32),
                                             (1, "hybrid", 32),
                                             (2, "hybrid", 32),
                                             (3, "hybrid", 32),
                                             (4, "hybrid", 32),
                                             (5, "hybrid", 32),
                                             (0, "forward", 32),
                                             (1, "forward", 32),
                                             (2, "forward", 32),
                                             (3, "forward", 32),
                                             (4, "forward", 32),
                                             (5, "forward", 32),
                                             (0, "ibp", 32),
                                             (1, "ibp", 32),
                                             (2, "ibp", 32),
                                             (3, "ibp", 32),
                                             (4, "ibp", 32),
                                             (5, "ibp", 32),
                                             (0, "hybrid", 64),
                                             (1, "hybrid", 64),
                                             (2, "hybrid", 64),
                                             (3, "hybrid", 64),
                                             (4, "hybrid", 64),
                                             (5, "hybrid", 64),
                                             (0, "forward", 64),
                                             (1, "forward", 64),
                                             (2, "forward", 64),
                                             (3, "forward", 64),
                                             (4, "forward", 64),
                                             (5, "forward", 64),
                                             (0, "ibp", 64),
                                             (1, "ibp", 64),
                                             (2, "ibp", 64),
                                             (3, "ibp", 64),
                                             (4, "ibp", 64),
                                             (5, "ibp", 64),
                                             (0, "hybrid", 16),
                                             (1, "hybrid", 16),
                                             (2, "hybrid", 16),
                                             (3, "hybrid", 16),
                                             (4, "hybrid", 16),
                                             (5, "hybrid", 16),
                                             (0, "forward", 16),
                                             (1, "forward", 16),
                                             (2, "forward", 16),
                                             (3, "forward", 16),
                                             (4, "forward", 16),
                                             (5, "forward", 16),
                                             (0, "ibp", 16),
                                             (1, "ibp", 16),
                                             (2, "ibp", 16),
                                             (3, "ibp", 16),
                                             (4, "ibp", 16),
                                             (5, "ibp", 16),
                                             ])
def test_softplus_backward_1D_box(n, mode, floatx):

    K.set_floatx('float{}'.format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs

    x_, y_, z_, u_c_, W_u_, B_u_, l_c_, W_l_, B_l_ = inputs_

    if mode =="hybrid":
        input_mode = inputs[2:]
        output = softplus_(input_mode, dc_decomp=False, mode=mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode=="forward":
        input_mode=[z,  W_u, b_u, W_l, b_l]
        output = softplus_(input_mode, dc_decomp=False, mode=mode)
        z_0, _, _, _, _ = output
    if mode =="ibp":
        input_mode=[u_c,l_c]
        output = softplus_(input_mode, dc_decomp=False, mode=mode)
        z_0 = z_
        u_c_0, l_c_0 = output

    w_out = Input((1, 1), dtype='float{}'.format(floatx))
    b_out = Input((1,), dtype='float{}'.format(floatx))

    #backward_softplus_(input_mode, w_out, b_out, w_out, b_out, mode=mode)
    #w_out_u, b_out_u, w_out_l, b_out_l = backward_softplus_(input_mode, w_out, b_out, w_out, b_out, mode=mode)



    f_relu = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
    output_ = f_relu(inputs_ + [np.ones((len(x_), 1, 1)), np.zeros((len(x_), 1))])
    w_u_, b_u_, w_l_, b_l_ = output_

    w_u_b = np.sum(np.maximum(w_u_, 0) * W_u_ + np.minimum(w_u_, 0) * W_l_, 1)[:, :, None]
    b_u_b = (
        b_u_
        + np.sum(np.maximum(w_u_, 0) * B_u_[:, :, None], 1)
        + np.sum(np.minimum(w_u_, 0) * B_l_[:, :, None], 1)
    )
    w_l_b = np.sum(np.maximum(w_l_, 0) * W_l_ + np.minimum(w_l_, 0) * W_u_, 1)[:, :, None]
    b_l_b = (
        b_l_
        + np.sum(np.maximum(w_l_[:, 0], 0) * B_l_[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * B_u_[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        w_u_b,
        b_u_b,
        None,
        w_l_b,
        b_l_b,
        "dense_{}".format(n),
        decimal=decimal
    )

    K.set_epsilon(eps)
    K.set_floatx('float32')
"""
