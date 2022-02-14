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
from decomon.backward_layers.activations import backward_relu, backward_sigmoid, backward_tanh


@pytest.mark.parametrize(
    "n, mode, previous, floatx",
    [
        (0, "hybrid", False, 32),
        (1, "hybrid", False, 32),
        (2, "hybrid", False, 32),
        (3, "hybrid", False, 32),
        (4, "hybrid", False, 32),
        (5, "hybrid", False, 32),
        (0, "forward", False, 32),
        (1, "forward", False, 32),
        (2, "forward", False, 32),
        (3, "forward", False, 32),
        (4, "forward", False, 32),
        (5, "forward", False, 32),
        (0, "ibp", False, 32),
        (1, "ibp", False, 32),
        (2, "ibp", False, 32),
        (3, "ibp", False, 32),
        (4, "ibp", False, 32),
        (5, "ibp", False, 32),
        (0, "hybrid", False, 64),
        (1, "hybrid", False, 64),
        (2, "hybrid", False, 64),
        (3, "hybrid", False, 64),
        (4, "hybrid", False, 64),
        (5, "hybrid", False, 64),
        (0, "forward", False, 64),
        (1, "forward", False, 64),
        (2, "forward", False, 64),
        (3, "forward", False, 64),
        (4, "forward", False, 64),
        (5, "forward", False, 64),
        (0, "ibp", False, 64),
        (1, "ibp", False, 64),
        (2, "ibp", False, 64),
        (3, "ibp", False, 64),
        (4, "ibp", False, 64),
        (5, "ibp", False, 64),
        (0, "hybrid", False, 16),
        (1, "hybrid", False, 16),
        (2, "hybrid", False, 16),
        (3, "hybrid", False, 16),
        (4, "hybrid", False, 16),
        (5, "hybrid", False, 16),
        (0, "forward", False, 16),
        (1, "forward", False, 16),
        (2, "forward", False, 16),
        (3, "forward", False, 16),
        (4, "forward", False, 16),
        (5, "forward", False, 16),
        (0, "ibp", False, 16),
        (1, "ibp", False, 16),
        (2, "ibp", False, 16),
        (3, "ibp", False, 16),
        (4, "ibp", False, 16),
        (5, "ibp", False, 16),
        (0, "hybrid", True, 32),
        (1, "hybrid", True, 32),
        (2, "hybrid", True, 32),
        (3, "hybrid", True, 32),
        (4, "hybrid", True, 32),
        (5, "hybrid", True, 32),
        (0, "forward", True, 32),
        (1, "forward", True, 32),
        (2, "forward", True, 32),
        (3, "forward", True, 32),
        (4, "forward", True, 32),
        (5, "forward", True, 32),
        (0, "ibp", True, 32),
        (1, "ibp", True, 32),
        (2, "ibp", True, 32),
        (3, "ibp", True, 32),
        (4, "ibp", True, 32),
        (5, "ibp", True, 32),
        (0, "hybrid", True, 64),
        (1, "hybrid", True, 64),
        (2, "hybrid", True, 64),
        (3, "hybrid", True, 64),
        (4, "hybrid", True, 64),
        (5, "hybrid", True, 64),
        (0, "forward", True, 64),
        (1, "forward", True, 64),
        (2, "forward", True, 64),
        (3, "forward", True, 64),
        (4, "forward", True, 64),
        (5, "forward", True, 64),
        (0, "ibp", True, 64),
        (1, "ibp", True, 64),
        (2, "ibp", True, 64),
        (3, "ibp", True, 64),
        (4, "ibp", True, 64),
        (5, "ibp", True, 64),
        (0, "hybrid", True, 16),
        (1, "hybrid", True, 16),
        (2, "hybrid", True, 16),
        (3, "hybrid", True, 16),
        (4, "hybrid", True, 16),
        (5, "hybrid", True, 16),
        (0, "forward", True, 16),
        (1, "forward", True, 16),
        (2, "forward", True, 16),
        (3, "forward", True, 16),
        (4, "forward", True, 16),
        (5, "forward", True, 16),
        (0, "ibp", True, 16),
        (1, "ibp", True, 16),
        (2, "ibp", True, 16),
        (3, "ibp", True, 16),
        (4, "ibp", True, 16),
        (5, "ibp", True, 16),
    ],
)
def test_activation_relu_backward_1D_box(n, mode, previous, floatx):

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

    w_out = Input((1, 1))
    b_out = Input((1))

    # backward_relu_(input_mode, w_out, b_out, w_out, b_out, mode=mode)
    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = backward_relu(
            input_mode + [w_out, b_out, w_out, b_out], mode=mode, previous=previous
        )
        f_relu = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_relu(inputs_ + [np.ones((len(x_), 1, 1)), np.zeros((len(x_), 1))])
    else:
        w_out_u, b_out_u, w_out_l, b_out_l = backward_relu(input_mode, mode=mode, previous=previous)
        f_relu = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_relu(inputs_)

    w_u_, b_u_, w_l_, b_l_ = output_

    w_u_b = np.sum(np.maximum(w_u_, 0) * W_u_ + np.minimum(w_u_, 0) * W_l_, 1)[:, :, None]
    b_u_b = b_u_ + np.sum(np.maximum(w_u_, 0) * B_u_[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * B_l_[:, :, None], 1)
    w_l_b = np.sum(np.maximum(w_l_, 0) * W_l_ + np.minimum(w_l_, 0) * W_u_, 1)[:, :, None]
    b_l_b = b_l_ + np.sum(np.maximum(w_l_, 0) * B_l_[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * B_u_[:, :, None], 1)

    assert_output_properties_box_linear(
        x_, None, z_[:, 0], z_[:, 1], None, w_u_b, b_u_b, None, w_l_b, b_l_b, "dense_{}".format(n), decimal=decimal
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")
