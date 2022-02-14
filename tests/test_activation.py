# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box,
    assert_output_properties_box_nodc,
    get_tensor_decomposition_multid_box,
    get_standard_values_multid_box,
)
import tensorflow.python.keras.backend as K

from decomon.layers.activations import sigmoid, tanh, softsign, softplus


@pytest.mark.parametrize(
    "n, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_sigmoid_1D_box(n, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-4)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs  # tensors
    (
        x_0,
        y_0,
        z_0,
        u_c_0,
        W_u_0,
        b_u_0,
        l_c_0,
        W_l_0,
        b_l_0,
    ) = inputs_  # numpy values

    if mode == "hybrid":
        output = sigmoid(inputs[2:], dc_decomp=False, mode=mode)
    if mode == "forward":
        output = sigmoid([z, W_u, b_u, W_l, b_l], dc_decomp=False, mode=mode)
    if mode == "ibp":
        output = sigmoid([u_c, l_c], dc_decomp=False, mode=mode)

    f_sigmoid = K.function(inputs[2:], output)
    f_ref = K.function(inputs, K.sigmoid(y))
    y_ = f_ref(inputs_)
    z_ = z_0
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_sigmoid(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_sigmoid(inputs_[2:])
        u_c_, l_c_ = [None] * 2
    if mode == "ibp":
        u_c_, l_c_ = f_sigmoid(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_0,
        y_,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "sigmoid_{}".format(n),
        decimal=decimal,
    )

    K.set_floatx("float32")
    K.set_epsilon(eps)


## tanh
@pytest.mark.parametrize(
    "n, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_tanh_1D_box(n, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-4)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs  # tensors
    (
        x_0,
        y_0,
        z_0,
        u_c_0,
        W_u_0,
        b_u_0,
        l_c_0,
        W_l_0,
        b_l_0,
    ) = inputs_  # numpy values

    if mode == "hybrid":
        output = tanh(inputs[2:], dc_decomp=False, mode=mode)
    if mode == "forward":
        output = tanh([z, W_u, b_u, W_l, b_l], dc_decomp=False, mode=mode)
    if mode == "ibp":
        output = tanh([u_c, l_c], dc_decomp=False, mode=mode)

    f_tanh = K.function(inputs[2:], output)
    f_ref = K.function(inputs, K.tanh(y))
    y_ = f_ref(inputs_)
    z_ = z_0
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_tanh(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_tanh(inputs_[2:])
        u_c_, l_c_ = [None] * 2
    if mode == "ibp":
        u_c_, l_c_ = f_tanh(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_0,
        y_,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "sigmoid_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


## softsign
@pytest.mark.parametrize(
    "n, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_softsign_1D_box(n, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 4
    if floatx == 16:
        K.set_epsilon(1e-4)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs  # tensors
    (
        x_0,
        y_0,
        z_0,
        u_c_0,
        W_u_0,
        b_u_0,
        l_c_0,
        W_l_0,
        b_l_0,
    ) = inputs_  # numpy values

    if mode == "hybrid":
        output = softsign(inputs[2:], dc_decomp=False, mode=mode)
    if mode == "forward":
        output = softsign([z, W_u, b_u, W_l, b_l], dc_decomp=False, mode=mode)
    if mode == "ibp":
        output = softsign([u_c, l_c], dc_decomp=False, mode=mode)

    f_softsign = K.function(inputs[2:], output)
    f_ref = K.function(inputs, K.softsign(y))
    y_ = f_ref(inputs_)
    z_ = z_0

    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_softsign(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_softsign(inputs_[2:])
        u_c_, l_c_ = [None] * 2
    if mode == "ibp":
        u_c_, l_c_ = f_softsign(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_0,
        y_,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "sigmoid_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


"""
@pytest.mark.parametrize("n, mode, floatx", [(0, "hybrid", 32), (1, "hybrid", 32), (2, 'hybrid', 32), \
                                        (3, 'hybrid', 32), (4, 'hybrid', 32), (5, 'hybrid', 32), (6, 'hybrid', 32), (7, 'hybrid', 32),\
                                        (8, 'hybrid', 32), (9, 'hybrid', 32),
                                        (0, "forward", 32), (1, "forward", 32), (2, 'forward', 32), \
                                        (3, 'forward', 32), (4, 'forward', 32), (5, 'forward', 32), (6, 'forward', 32), (7, 'forward', 32),\
                                        (8, 'forward', 32), (9, 'forward', 32),
                                        (0, "ibp", 32), (1, "ibp", 32), (2, 'ibp', 32), \
                                        (3, 'ibp', 32), (4, 'ibp', 32), (5, 'ibp', 32), (6, 'ibp', 32), (7, 'ibp', 32),\
                                        (8, 'ibp', 32), (9, 'ibp', 32),
                                        (0, "hybrid", 64), (1, "hybrid", 64), (2, 'hybrid', 64), \
                                        (3, 'hybrid', 64), (4, 'hybrid', 64), (5, 'hybrid', 64), (6, 'hybrid', 64), (7, 'hybrid', 64),\
                                        (8, 'hybrid', 64), (9, 'hybrid', 64),
                                        (0, "forward", 64), (1, "forward", 64), (2, 'forward', 64), \
                                        (3, 'forward', 64), (4, 'forward', 64), (5, 'forward', 64), (6, 'forward', 64), (7, 'forward', 64),\
                                        (8, 'forward', 64), (9, 'forward', 64),
                                        (0, "ibp", 64), (1, "ibp", 64), (2, 'ibp', 64), \
                                        (3, 'ibp', 64), (4, 'ibp', 64), (5, 'ibp', 64), (6, 'ibp', 64), (7, 'ibp', 64),\
                                        (8, 'ibp', 64), (9, 'ibp', 64),
                                        (0, "hybrid", 16), (1, "hybrid", 16), (2, 'hybrid', 16), \
                                        (3, 'hybrid', 16), (4, 'hybrid', 16), (5, 'hybrid', 16), (6, 'hybrid', 16), (7, 'hybrid', 16),\
                                        (8, 'hybrid', 16), (9, 'hybrid', 16),
                                        (0, "forward", 16), (1, "forward", 16), (2, 'forward', 16), \
                                        (3, 'forward', 16), (4, 'forward', 16), (5, 'forward', 16), (6, 'forward', 16), (7, 'forward', 16),\
                                        (8, 'forward', 16), (9, 'forward', 16),
                                        (0, "ibp", 16), (1, "ibp", 16), (2, 'ibp', 16), \
                                        (3, 'ibp', 16), (4, 'ibp', 16), (5, 'ibp', 16), (6, 'ibp', 16), (7, 'ibp', 16),\
                                        (8, 'ibp', 16), (9, 'ibp', 16),
                                        ])
def test_softplus_1D_box(n, mode, floatx):

    K.set_floatx('float{}'.format(floatx))
    eps = K.epsilon()
    decimal = 4
    if floatx == 16:
        K.set_epsilon(1e-4)
        decimal = 2
    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs  # tensors
    (
        x_0,
        y_0,
        z_0,
        u_c_0,
        W_u_0,
        b_u_0,
        l_c_0,
        W_l_0,
        b_l_0,
    ) = inputs_  # numpy values

    if mode == 'hybrid':
        output = softplus(inputs[2:], dc_decomp=False, mode=mode)
    if mode == 'forward':
        output = softplus([z, W_u, b_u, W_l, b_l], dc_decomp=False, mode=mode)
    if mode == 'ibp':
        output = softplus([u_c, l_c], dc_decomp=False, mode=mode)


    f_softplus = K.function(inputs[2:], output)
    f_ref = K.function(inputs, K.softplus(y))
    y_ = f_ref(inputs_)
    z_ = z_0


    if mode == 'hybrid':
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_softplus(inputs_[2:])
    if mode == 'forward':
        z_, w_u_, b_u_, w_l_, b_l_ = f_softplus(inputs_[2:])
        u_c_, l_c_ = [None] * 2
    if mode == 'ibp':
        u_c_, l_c_ = f_softplus(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_0, y_, None, None, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "sigmoid_{}".format(n), decimal=decimal
    )
    
    K.set_floatx('float32')
    K.set_epsilon(eps)
"""  # TODO check
