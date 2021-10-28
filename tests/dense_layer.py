# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from decomon.layers.decomon_layers import DecomonDense, to_monotonic
from tensorflow.keras.layers import Dense
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box,
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_tensor_decomposition_multid_box,
)
import tensorflow.python.keras.backend as K
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize(
    "n, activation, n_subgrad",
    [
        (0, "relu", 0),
        (1, "relu", 0),
        (2, "relu", 0),
        (3, "relu", 0),
        (4, "relu", 0),
        (5, "relu", 0),
        (6, "relu", 0),
        (7, "relu", 0),
        (8, "relu", 0),
        (9, "relu", 0),
        (0, "linear", 0),
        (1, "linear", 0),
        (2, "linear", 0),
        (3, "linear", 0),
        (4, "linear", 0),
        (5, "linear", 0),
        (6, "linear", 0),
        (7, "linear", 0),
        (8, "linear", 0),
        (9, "linear", 0),
        (0, None, 0),
        (1, None, 0),
        (2, None, 0),
        (3, None, 0),
        (4, None, 0),
        (5, None, 0),
        (6, None, 0),
        (7, None, 0),
        (8, None, 0),
        (9, None, 0),
        (0, "relu", 1),
        (1, "relu", 1),
        (2, "relu", 1),
        (3, "relu", 1),
        (4, "relu", 1),
        (5, "relu", 1),
        (6, "relu", 1),
        (7, "relu", 1),
        (8, "relu", 1),
        (9, "relu", 1),
        (0, "linear", 1),
        (1, "linear", 1),
        (2, "linear", 1),
        (3, "linear", 1),
        (4, "linear", 1),
        (5, "linear", 1),
        (6, "linear", 1),
        (7, "linear", 1),
        (8, "linear", 1),
        (9, "linear", 1),
        (0, None, 1),
        (1, None, 1),
        (2, None, 1),
        (3, None, 1),
        (4, None, 1),
        (5, None, 1),
        (6, None, 1),
        (7, None, 1),
        (8, None, 1),
        (9, None, 1),
        (0, "relu", 5),
        (1, "relu", 5),
        (2, "relu", 5),
        (3, "relu", 5),
        (4, "relu", 5),
        (5, "relu", 5),
        (6, "relu", 5),
        (7, "relu", 5),
        (8, "relu", 5),
        (9, "relu", 5),
        (0, "linear", 5),
        (1, "linear", 5),
        (2, "linear", 5),
        (3, "linear", 5),
        (4, "linear", 5),
        (5, "linear", 5),
        (6, "linear", 5),
        (7, "linear", 5),
        (8, "linear", 5),
        (9, "linear", 5),
        (0, None, 5),
        (1, None, 5),
        (2, None, 5),
        (3, None, 5),
        (4, None, 5),
        (5, None, 5),
        (6, None, 5),
        (7, None, 5),
        (8, None, 5),
        (9, None, 5),
    ],
)
def test_DecomonDense_1D_box(n, activation, n_subgrad):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=True)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output = monotonic_dense(inputs[1:])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

    f_dense = K.function(inputs[1:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_output_properties_box(
        x,
        y_ref,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
    )
    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
    )

    monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[1:])

    y_ref = f_ref(inputs_)

    assert_output_properties_box(
        x,
        y_ref,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
    )
    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
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
def test_DecomonDense_multiD_box(odd, activation, n_subgrad):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=True)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output = monotonic_dense(inputs[1:])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    f_dense = K.function(inputs[1:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_output_properties_box(
        x,
        y_ref,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_multid_{}".format(odd),
    )
    assert np.allclose(y_ref, y_)

    monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_output_properties_box(
        x,
        y_ref,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_multid_{}".format(odd),
    )
    assert np.allclose(y_ref, y_)


@pytest.mark.parametrize(
    "n, activation, n_subgrad",
    [
        (0, "relu", 0),
        (1, "relu", 0),
        (2, "relu", 0),
        (3, "relu", 0),
        (4, "relu", 0),
        (5, "relu", 0),
        (6, "relu", 0),
        (7, "relu", 0),
        (8, "relu", 0),
        (9, "relu", 0),
        (0, "linear", 0),
        (1, "linear", 0),
        (2, "linear", 0),
        (3, "linear", 0),
        (4, "linear", 0),
        (5, "linear", 0),
        (6, "linear", 0),
        (7, "linear", 0),
        (8, "linear", 0),
        (9, "linear", 0),
        (0, None, 0),
        (1, None, 0),
        (2, None, 0),
        (3, None, 0),
        (4, None, 0),
        (5, None, 0),
        (6, None, 0),
        (7, None, 0),
        (8, None, 0),
        (9, None, 0),
        (0, "relu", 1),
        (1, "relu", 1),
        (2, "relu", 1),
        (3, "relu", 1),
        (4, "relu", 1),
        (5, "relu", 1),
        (6, "relu", 1),
        (7, "relu", 1),
        (8, "relu", 1),
        (9, "relu", 1),
        (0, "linear", 1),
        (1, "linear", 1),
        (2, "linear", 1),
        (3, "linear", 1),
        (4, "linear", 1),
        (5, "linear", 1),
        (6, "linear", 1),
        (7, "linear", 1),
        (8, "linear", 1),
        (9, "linear", 1),
        (0, None, 1),
        (1, None, 1),
        (2, None, 1),
        (3, None, 1),
        (4, None, 1),
        (5, None, 1),
        (6, None, 1),
        (7, None, 1),
        (8, None, 1),
        (9, None, 1),
        (0, "relu", 5),
        (1, "relu", 5),
        (2, "relu", 5),
        (3, "relu", 5),
        (4, "relu", 5),
        (5, "relu", 5),
        (6, "relu", 5),
        (7, "relu", 5),
        (8, "relu", 5),
        (9, "relu", 5),
        (0, "linear", 5),
        (1, "linear", 5),
        (2, "linear", 5),
        (3, "linear", 5),
        (4, "linear", 5),
        (5, "linear", 5),
        (6, "linear", 5),
        (7, "linear", 5),
        (8, "linear", 5),
        (9, "linear", 5),
        (0, None, 5),
        (1, None, 5),
        (2, None, 5),
        (3, None, 5),
        (4, None, 5),
        (5, None, 5),
        (6, None, 5),
        (7, None, 5),
        (8, None, 5),
        (9, None, 5),
    ],
)
def test_DecomonDense_1D_to_monotonic_box(n, activation, n_subgrad):

    dense_ref = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output_ref = dense_ref(inputs[1])
    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]

    monotonic_dense = to_monotonic(dense_ref, (2, input_dim), dc_decomp=True)

    W_, bias = monotonic_dense.get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    output = monotonic_dense(inputs[1:])

    f_dense = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_to_monotonic_{}".format(n),
        decimal=5,
    )
    assert np.allclose(y_ref, y_)


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
def test_DecomonDense_multiD_to_monotonic_box(odd, activation, n_subgrad):

    dense_ref = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=True)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=True)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output_ref = dense_ref(inputs[1])
    W_0, b_0 = dense_ref.get_weights()
    if odd == 1:

        W_0[0] = -0.1657672
        W_0[1] = -0.2613032
        W_0[2] = 0.08437371
        dense_ref.set_weights([W_0, b_0])

    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]

    y_ = f_ref(inputs_)

    monotonic_dense = to_monotonic(dense_ref, (2, input_dim), dc_decomp=True)

    W_, bias = monotonic_dense.get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    output = monotonic_dense(inputs[1:])

    f_dense = K.function(inputs[1:], output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[1:])

    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_multid_to_monotonic{}".format(odd),
    )


"""

# DC DECOMP = FALSE
"""


@pytest.mark.parametrize(
    "n, activation, n_subgrad",
    [
        (0, "relu", 0),
        (1, "relu", 0),
        (2, "relu", 0),
        (3, "relu", 0),
        (4, "relu", 0),
        (5, "relu", 0),
        (6, "relu", 0),
        (7, "relu", 0),
        (8, "relu", 0),
        (9, "relu", 0),
        (0, "linear", 0),
        (1, "linear", 0),
        (2, "linear", 0),
        (3, "linear", 0),
        (4, "linear", 0),
        (5, "linear", 0),
        (6, "linear", 0),
        (7, "linear", 0),
        (8, "linear", 0),
        (9, "linear", 0),
        (0, None, 0),
        (1, None, 0),
        (2, None, 0),
        (3, None, 0),
        (4, None, 0),
        (5, None, 0),
        (6, None, 0),
        (7, None, 0),
        (8, None, 0),
        (9, None, 0),
        (0, "relu", 1),
        (1, "relu", 1),
        (2, "relu", 1),
        (3, "relu", 1),
        (4, "relu", 1),
        (5, "relu", 1),
        (6, "relu", 1),
        (7, "relu", 1),
        (8, "relu", 1),
        (9, "relu", 1),
        (0, "linear", 1),
        (1, "linear", 1),
        (2, "linear", 1),
        (3, "linear", 1),
        (4, "linear", 1),
        (5, "linear", 1),
        (6, "linear", 1),
        (7, "linear", 1),
        (8, "linear", 1),
        (9, "linear", 1),
        (0, None, 1),
        (1, None, 1),
        (2, None, 1),
        (3, None, 1),
        (4, None, 1),
        (5, None, 1),
        (6, None, 1),
        (7, None, 1),
        (8, None, 1),
        (9, None, 1),
        (0, "relu", 5),
        (1, "relu", 5),
        (2, "relu", 5),
        (3, "relu", 5),
        (4, "relu", 5),
        (5, "relu", 5),
        (6, "relu", 5),
        (7, "relu", 5),
        (8, "relu", 5),
        (9, "relu", 5),
        (0, "linear", 5),
        (1, "linear", 5),
        (2, "linear", 5),
        (3, "linear", 5),
        (4, "linear", 5),
        (5, "linear", 5),
        (6, "linear", 5),
        (7, "linear", 5),
        (8, "linear", 5),
        (9, "linear", 5),
        (0, None, 5),
        (1, None, 5),
        (2, None, 5),
        (3, None, 5),
        (4, None, 5),
        (5, None, 5),
        (6, None, 5),
        (7, None, 5),
        (8, None, 5),
        (9, None, 5),
    ],
)
def test_DecomonDense_1D_box_nodc(n, activation, n_subgrad):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_dense(inputs[1:])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

    f_dense = K.function(inputs[1:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)
    assert_almost_equal(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")


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
def test_DecomonDense_multiD_to_monotonic_box_nodc(odd, activation, n_subgrad):

    dense_ref = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output_ref = dense_ref(inputs[1])
    W_0, b_0 = dense_ref.get_weights()
    if odd == 0:
        W_0[0] = 1.037377
        W_0[1] = -0.7575816

        dense_ref.set_weights([W_0, b_0])
    if odd == 1:

        W_0[0] = -0.1657672
        W_0[1] = -0.2613032
        W_0[2] = 0.08437371
        dense_ref.set_weights([W_0, b_0])

    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]

    y_ref = f_ref(inputs_)

    monotonic_dense = to_monotonic(dense_ref, (2, input_dim), dc_decomp=False)

    W_, bias = monotonic_dense.get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    output = monotonic_dense(inputs[1:])

    f_dense = K.function(inputs[1:], output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[1:])

    assert_almost_equal(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")


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
def test_DecomonDense_multiD_box_dc(odd, activation, n_subgrad):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_dense(inputs[1:])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    f_dense = K.function(inputs[1:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")

    monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")
