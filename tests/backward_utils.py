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


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_relu_backward_1D_box(n):

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = relu_(inputs[1:], dc_decomp=False)
    y_0, z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    w_out_u, b_out_u, w_out_l, b_out_l = backward_relu_(inputs[1:], w_out, b_out, w_out, b_out)

    f_relu = K.function(inputs + [w_out, b_out], [y_0, z_0, u_c_0, w_out_u, b_out_u, l_c_0, w_out_l, b_out_l])
    output_ = f_relu(inputs_ + [np.ones((len(x), 1, 1, 1)), np.zeros((len(x), 1, 1))])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_

    w_u_b = np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None]
    b_u_b = (
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1)
    )
    w_l_b = np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None]
    b_l_b = (
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x,
        np.maximum(y_, 0.0),
        z_[:, 0],
        z_[:, 1],
        np.maximum(u_c_, 0.0),
        w_u_b,
        b_u_b,
        np.maximum(l_c_, 0.0),
        w_l_b,
        b_l_b,
        "dense_{}".format(n),
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_softplus_backward_1D_box(n):

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = softplus_(inputs[1:], dc_decomp=False)
    y_0, z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    w_out_u, b_out_u, w_out_l, b_out_l = backward_softplus_(inputs[1:], w_out, b_out, w_out, b_out)

    f_relu = K.function(inputs + [w_out, b_out], [y_0, z_0, u_c_0, w_out_u, b_out_u, l_c_0, w_out_l, b_out_l])
    output_ = f_relu(inputs_ + [np.ones((len(x), 1, 1, 1)), np.zeros((len(x), 1, 1))])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_

    w_u_b = np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None]
    b_u_b = (
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1)
    )
    w_l_b = np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None]
    b_l_b = (
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x,
        np.maximum(y_, 0.0),
        z_[:, 0],
        z_[:, 1],
        np.maximum(u_c_, 0.0),
        w_u_b,
        b_u_b,
        np.maximum(l_c_, 0.0),
        w_l_b,
        b_l_b,
        "dense_{}".format(n),
    )


@pytest.mark.parametrize("n_0, n_1", [(0, 3), (1, 4), (2, 5)])
def test_add_backward_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    back_bounds_0, back_bounds_1 = backward_add(inputs_0[1:], inputs_1[1:], w_out, b_out, w_out, b_out)
    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u_0 + np.minimum(w_u_0_[:, 0], 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l_0 + np.minimum(w_l_0_[:, 0], 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
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
    )


@pytest.mark.parametrize("n_0, n_1", [(0, 3), (1, 4), (2, 5)])
# @pytest.mark.parametrize("n_0, n_1", [(0, 1)])
def test_substract_backward_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    back_bounds_0, back_bounds_1 = backward_substract(inputs_0[1:], inputs_1[1:], w_out, b_out, w_out, b_out)
    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)

    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    f_ref = K.function(inputs_0 + inputs_1, substract(inputs_0[1:], inputs_1[1:]))

    y_, _, u_, _, _, l_, _, _ = f_ref(inputs_0_ + inputs_1_)

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u_0 + np.minimum(w_u_0_[:, 0], 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l_0 + np.minimum(w_l_0_[:, 0], 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
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
    )


@pytest.mark.parametrize("n_0, n_1", [(0, 3), (1, 4), (2, 5)])
def test_maximum_backward_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    back_bounds_0, back_bounds_1 = backward_maximum(inputs_0[1:], inputs_1[1:], w_out, b_out, w_out, b_out)

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u_0 + np.minimum(w_u_0_[:, 0], 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l_0 + np.minimum(w_l_0_[:, 0], 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
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
    )


"""
@pytest.mark.parametrize("odd", [1])
def test_max_backward_multiD_box(odd):

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x_, y_, z_, u_c_, W_u, b_u, l_c_, W_l, b_l = inputs_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    back_bounds = backward_max_(inputs[1:], w_out, b_out, w_out, b_out)

    f_add = K.function(inputs + [w_out, b_out], back_bounds)
    output_ = f_add(inputs_ + [np.ones((len(x_), 1, 1, 1)), np.zeros((len(x_), 1, 1))])

    w_u_0_, b_u_0_, w_l_0_, b_l_0_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u + np.minimum(w_u_0_[:, 0], 0) * W_l, 1)[:, :, None]
    import pdb; pdb.set_trace()
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l + np.minimum(w_l_0_[:, 0], 0) * W_u, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u[:, :, None], 1)
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
"""
