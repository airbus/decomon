# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from numpy.testing import assert_almost_equal
from tensorflow.keras.layers import Dense

from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import DecomonDense, to_decomon


def test_DecomonDense_1D_box(n, activation, mode, shared, floatx, helpers):

    K.set_floatx(f"float{floatx}")
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    monotonic_dense = DecomonDense(
        1, use_bias=True, activation=activation, dc_decomp=True, mode=mode, shared=shared, dtype=K.floatx()
    )

    ref_dense = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_dense(inputs[1])
    monotonic_dense.share_weights(ref_dense)

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = monotonic_dense(inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = monotonic_dense([z, W_u, b_u, W_l, b_l, h, g])
    elif mode == ForwardMode.IBP:
        output = monotonic_dense([u_c, l_c, h, g])
    else:
        raise ValueError("Unknown mode.")

    W_, bias = monotonic_dense.get_weights()
    if not shared:
        monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    y_ref = f_ref(inputs_)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
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
            f"dense_{n}",
            decimal=decimal,
        )

    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        u_c_ = None
        l_c_ = None
        helpers.assert_output_properties_box(
            x_,
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
            f"dense_{n}",
            decimal=decimal,
        )

    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            f"dense_{n}",
            decimal=decimal,
        )
    else:
        raise ValueError("Unknown mode.")

    if not shared:
        monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    y_ref = f_ref(inputs_)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
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
            f"dense_{n}",
            decimal=decimal,
        )

    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        u_c_ = None
        l_c_ = None
        helpers.assert_output_properties_box(
            x_,
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
            f"dense_{n}",
            decimal=decimal,
        )

    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            f"dense_{n}",
            decimal=decimal,
        )
    else:
        raise ValueError("Unknown mode.")

    K.set_epsilon(eps)
    K.set_floatx("float32")


def test_DecomonDense_multiD_box(odd, activation, mode, helpers):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=True, mode=mode, dtype=K.floatx())
    ref_dense = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_ = helpers.get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = monotonic_dense(inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = monotonic_dense([z, W_u, b_u, W_l, b_l, h, g])
    elif mode == ForwardMode.IBP:
        output = monotonic_dense([u_c, l_c, h, g])
    else:
        raise ValueError("Unknown mode.")

    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))
    y_ref = f_ref(inputs_)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
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
            f"dense_multid_{odd}",
        )
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        u_c_ = None
        l_c_ = None
        helpers.assert_output_properties_box(
            x_,
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
            f"dense_multid_{odd}",
        )
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "dense_multid_{}".format(odd),
        )
    else:
        raise ValueError("Unknown mode.")

    monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    y_ref = f_ref(inputs_)

    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
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
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        u_c_ = None
        l_c_ = None
        helpers.assert_output_properties_box(
            x_,
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
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "dense_multid_{}".format(odd),
        )
    else:
        raise ValueError("Unknown mode.")


def test_DecomonDense_1D_to_monotonic_box(n, activation, mode, shared, helpers):

    dense_ref = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    z_ = inputs_[2]
    x_ = inputs_[0]

    output_ref = dense_ref(inputs[1])
    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]
    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        ibp = True
        affine = True
    elif mode == ForwardMode.AFFINE:
        ibp = False
        affine = True
    elif mode == ForwardMode.IBP:
        ibp = True
        affine = False
    else:
        raise ValueError("Unknown mode.")

    monotonic_dense = to_decomon(
        dense_ref, input_dim, dc_decomp=True, ibp=ibp, affine=affine, shared=shared
    )  # ATTENTION: it will be a list

    W_, bias = monotonic_dense[0].get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    if mode == ForwardMode.HYBRID:
        output = monotonic_dense[0](inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = monotonic_dense[0]([z, W_u, b_u, W_l, b_l, h, g])
    elif mode == ForwardMode.IBP:
        output = monotonic_dense[0]([u_c, l_c, h, g])
    else:
        raise ValueError("Unknown mode.")

    if len(monotonic_dense) > 1:
        output = monotonic_dense[1](output)

    f_dense = K.function(inputs[2:], output)
    y_ref = f_ref(inputs_)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
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
            "dense_to_monotonic_{}".format(n),
            decimal=5,
        )

    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            None,
            w_u_,
            b_u_,
            None,
            w_l_,
            b_l_,
            "dense_to_monotonic_{}".format(n),
            decimal=5,
        )

    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "dense_to_monotonic_{}".format(n),
            decimal=5,
        )
    else:
        raise ValueError("Unknown mode.")


def test_DecomonDense_multiD_to_monotonic_box(odd, activation, mode, helpers):

    dense_ref = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=True)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=True)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

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

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        ibp = True
        affine = True
    elif mode == ForwardMode.AFFINE:
        ibp = False
        affine = True
    elif mode == ForwardMode.IBP:
        ibp = True
        affine = False
    else:
        raise ValueError("Unknown mode.")

    monotonic_dense = to_decomon(dense_ref, input_dim, dc_decomp=True, ibp=ibp, affine=affine)

    W_, bias = monotonic_dense[0].get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    if mode == ForwardMode.HYBRID:
        output = monotonic_dense[0](inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = monotonic_dense[0]([z, W_u, b_u, W_l, b_l, h, g])
    elif mode == ForwardMode.IBP:
        output = monotonic_dense[0]([u_c, l_c, h, g])
    else:
        raise ValueError("Unknown mode.")

    if len(monotonic_dense) > 1:
        output = monotonic_dense[1](output)

    f_dense = K.function(inputs[2:], output)

    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
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
            "dense_to_monotonic_{}".format(0),
            decimal=5,
        )

    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            None,
            w_u_,
            b_u_,
            None,
            w_l_,
            b_l_,
            "dense_to_monotonic_{}".format(0),
            decimal=5,
        )

    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "dense_to_monotonic_{}".format(0),
            decimal=5,
        )
    else:
        raise ValueError("Unknown mode.")


def test_DecomonDense_1D_box_nodc(n, activation, helpers):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_dense(inputs[2:])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    y_ref = f_ref(inputs_)
    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )


def test_DecomonDense_multiD_to_monotonic_box_nodc(odd, activation, mode, helpers):

    dense_ref = Dense(1, use_bias=True, activation=activation)

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
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

    monotonic_dense = to_decomon(dense_ref, input_dim, dc_decomp=False)

    W_, bias = monotonic_dense[0].get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    output = monotonic_dense[0](inputs[2:])
    if len(monotonic_dense) > 1:
        output = monotonic_dense[1](output)

    f_dense = K.function(inputs[2:], output)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])

    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )


def test_DecomonDense_multiD_box_dc(odd, activation, helpers):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_dense(inputs[2:])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    y_ref = f_ref(inputs_)

    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )

    monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    y_ref = f_ref(inputs_)

    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )
