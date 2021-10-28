# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box_nodc,
    get_tensor_decomposition_multid_box,
    get_standard_values_multid_box,
)
import tensorflow.python.keras.backend as K

from decomon.layers.activations import sigmoid, tanh, softsign, softplus


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_sigmoid_1D_box(n):

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

    output = sigmoid(inputs[1:], dc_decomp=False)
    f_sigmoid = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_sigmoid(inputs_[1:])

    assert_output_properties_box_nodc(
        x_0, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "sigmoid_{}".format(n)
    )


## tanh
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_tanh_1D_box(n):

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

    output = tanh(inputs[1:], dc_decomp=False)
    f_tanh = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_tanh(inputs_[1:])

    assert_output_properties_box_nodc(
        x_0, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "sigmoid_{}".format(n)
    )


## softsign
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_softsign_1D_box(n):

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

    output = softsign(inputs[1:], dc_decomp=False)
    f_softsign = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_softsign(inputs_[1:])

    assert_output_properties_box_nodc(
        x_0, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "sigmoid_{}".format(n), decimal=4
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_softplus_1D_box(n):

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

    output = softplus(inputs[1:], dc_decomp=False)
    f_softplus = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_softplus(inputs_[1:])

    assert_output_properties_box_nodc(
        x_0, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "softplus_{}".format(n)
    )
