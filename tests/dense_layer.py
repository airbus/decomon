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
def test_DecomonDense_1D_box(n, activation):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=True)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output = monotonic_dense(inputs[1:])
    ref_dense(inputs[1])

    W_pos, W_neg, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_pos), np.zeros_like(W_neg), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_pos), np.ones_like(bias)])

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

    monotonic_dense.set_weights([np.zeros_like(W_pos), -3 * np.ones_like(W_neg), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_pos), np.ones_like(bias)])
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
    "odd, activation", [(0, None), (1, None), (0, "linear"), (1, "linear"), (0, "relu"), (1, "relu")]
)
def test_DecomonDense_multiD_box(odd, activation):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=True)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output = monotonic_dense(inputs[1:])
    ref_dense(inputs[1])

    W_pos, W_neg, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_pos), np.zeros_like(W_neg), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_pos), np.ones_like(bias)])
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

    monotonic_dense.set_weights([np.zeros_like(W_pos), -3 * np.ones_like(W_neg), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_pos), np.ones_like(bias)])
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
def test_DecomonDense_1D_to_monotonic_box(n, activation):

    dense_ref = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output_ref = dense_ref(inputs[1])
    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]

    monotonic_dense = to_monotonic(dense_ref, (2, input_dim), dc_decomp=True)

    W_pos, W_neg, bias = monotonic_dense.get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_pos + W_neg, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(
        np.minimum(W_pos, -W_neg),
        np.zeros_like(W_pos),
        decimal=6,
        err_msg="wrong decomposition",
    )
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
    "odd, activation", [(0, None), (1, None), (0, "linear"), (1, "linear"), (0, "relu"), (1, "relu")]
)
def test_DecomonDense_multiD_to_monotonic_box(odd, activation):

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

    W_pos, W_neg, bias = monotonic_dense.get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_pos + W_neg, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(
        np.minimum(W_pos, -W_neg),
        np.zeros_like(W_pos),
        decimal=6,
        err_msg="wrong decomposition",
    )
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


# DC DECOMP = FALSE


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
def test_DecomonDense_1D_box_nodc(n, activation):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_dense(inputs[1:])
    ref_dense(inputs[1])

    W_pos, W_neg, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_pos), np.zeros_like(W_neg), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_pos), np.ones_like(bias)])

    f_dense = K.function(inputs[1:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)
    assert_almost_equal(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")


@pytest.mark.parametrize(
    "odd, activation", [(0, None), (1, None), (0, "linear"), (1, "linear"), (0, "relu"), (1, "relu")]
)
def test_DecomonDense_multiD_to_monotonic_box_nodc(odd, activation):

    dense_ref = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output_ref = dense_ref(inputs[1])
    W_0, b_0 = dense_ref.get_weights()
    if odd == 1:

        W_0[0] = -0.1657672
        W_0[1] = -0.2613032
        W_0[2] = 0.08437371
        dense_ref.set_weights([W_0, b_0])

    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]

    y_ref = f_ref(inputs_)

    monotonic_dense = to_monotonic(dense_ref, (2, input_dim), dc_decomp=False)

    W_pos, W_neg, bias = monotonic_dense.get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_pos + W_neg, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(
        np.minimum(W_pos, -W_neg),
        np.zeros_like(W_pos),
        decimal=6,
        err_msg="wrong decomposition",
    )
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    output = monotonic_dense(inputs[1:])

    f_dense = K.function(inputs[1:], output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[1:])
    assert_almost_equal(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")


@pytest.mark.parametrize(
    "odd, activation", [(0, None), (1, None), (0, "linear"), (1, "linear"), (0, "relu"), (1, "relu")]
)
def test_DecomonDense_multiD_box_dc(odd, activation):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_dense(inputs[1:])
    ref_dense(inputs[1])

    W_pos, W_neg, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_pos), np.zeros_like(W_neg), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_pos), np.ones_like(bias)])
    f_dense = K.function(inputs[1:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")

    monotonic_dense.set_weights([np.zeros_like(W_pos), -3 * np.ones_like(W_neg), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_pos), np.ones_like(bias)])
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")
