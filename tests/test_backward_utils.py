# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Input

from decomon.backward_layers.utils import (
    backward_add,
    backward_maximum,
    backward_relu_,
    backward_substract,
)
from decomon.layers.core import ForwardMode
from decomon.utils import relu_, substract


def add_op(x, y):
    return x + y


def substract_op(x, y):
    return x - y


def test_relu_backward_1D_box(n, mode, floatx, helpers):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs

    x_, y_, z_, u_c_, W_u_, B_u_, l_c_, W_l_, B_l_ = inputs_

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        input_mode = inputs[2:]
        output = relu_(input_mode, dc_decomp=False, mode=mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    elif mode == ForwardMode.AFFINE:
        input_mode = [z, W_u, b_u, W_l, b_l]
        output = relu_(input_mode, dc_decomp=False, mode=mode)
        z_0, _, _, _, _ = output
    elif mode == ForwardMode.IBP:
        input_mode = [u_c, l_c]
        output = relu_(input_mode, dc_decomp=False, mode=mode)
        z_0 = z_
        u_c_0, l_c_0 = output
    else:
        raise ValueError("Unknown mode.")

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

    helpers.assert_output_properties_box_linear(
        x_, None, z_[:, 0], z_[:, 1], None, w_u_b, b_u_b, None, w_l_b, b_l_b, "dense_{}".format(n), decimal=decimal
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "n_0, n_1",
    [
        (0, 3),
        (1, 4),
        (2, 5),
    ],
)
@pytest.mark.parametrize(
    "backward_func, tensor_op",
    [
        (backward_add, add_op),
        (backward_maximum, np.maximum),
        (backward_substract, substract_op),
    ],
)
def test_reduce_backward_1D_box(n_0, n_1, backward_func, tensor_op, mode, floatx, helpers):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs_0 = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = helpers.get_standard_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = helpers.get_standard_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1,), dtype=K.floatx())

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        input_tmp_0 = inputs_0[2:]
        input_tmp_1 = inputs_1[2:]
    elif mode == ForwardMode.IBP:
        input_tmp_0 = [inputs_0[3], inputs_0[6]]
        input_tmp_1 = [inputs_1[3], inputs_1[6]]
    elif mode == ForwardMode.AFFINE:
        input_tmp_0 = [inputs_0[2], inputs_0[4], inputs_0[5], inputs_0[7], inputs_0[8]]
        input_tmp_1 = [inputs_1[2], inputs_1[4], inputs_1[5], inputs_1[7], inputs_1[8]]
    else:
        raise ValueError("Unknown mode.")

    back_bounds_0, back_bounds_1 = backward_func(input_tmp_0, input_tmp_1, w_out, b_out, w_out, b_out, mode=mode)
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

    helpers.assert_output_properties_box_linear(
        x_0,
        tensor_op(y_0, y_1),
        z_0[:, 0],
        z_0[:, 1],
        tensor_op(u_c_0, u_c_1),
        w_u_b_0,
        b_u_b_0,
        tensor_op(l_c_0, l_c_1),
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
        decimal=decimal,
    )

    helpers.assert_output_properties_box_linear(
        x_1,
        tensor_op(y_0, y_1),
        z_1[:, 0],
        z_1[:, 1],
        tensor_op(u_c_0, u_c_1),
        w_u_b_1,
        b_u_b_1,
        tensor_op(l_c_0, l_c_1),
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
        decimal=decimal,
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")
