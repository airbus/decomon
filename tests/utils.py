# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box,
    get_tensor_decomposition_multid_box,
    get_standard_values_multid_box,
)
import tensorflow.python.keras.backend as K

from decomon.layers.utils import get_upper, get_lower, relu_, max_, maximum, add, minus, substract


@pytest.mark.parametrize("odd, floatx", [(0, 32), (0, 64), (0, 16)])
def test_get_upper_multi_box(odd, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-4)

    inputs = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)

    x, y, x_0, u_c, W_u, b_u, _, _, _, _, _ = inputs
    x_, y_, x_0_, u_c_, W_u_, b_u_, _, _, _, _, _ = inputs_

    # compute maximum
    x_min_ = x_0_[:, 0][:, :, None]
    x_max_ = x_0_[:, 1][:, :, None]

    upper_pred = np.sum(np.minimum(W_u_, 0) * x_min_ + np.maximum(W_u_, 0) * x_max_, 1) + b_u_

    upper = get_upper(x_0, W_u, b_u)

    f_upper = K.function([x_0, W_u, b_u], upper)

    upper_ = f_upper([x_0_, W_u_, b_u_])

    assert_allclose(upper_pred, upper_)
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n, floatx",
    [
        (0, 32),
        (1, 32),
        (2, 32),
        (3, 32),
        (4, 32),
        (5, 32),
        (6, 32),
        (7, 32),
        (8, 32),
        (9, 32),
        (0, 64),
        (1, 64),
        (2, 64),
        (3, 64),
        (4, 64),
        (5, 64),
        (6, 64),
        (7, 64),
        (8, 64),
        (9, 64),
        (0, 16),
        (1, 16),
        (2, 16),
        (3, 16),
        (4, 16),
        (5, 16),
        (6, 16),
        (7, 16),
        (8, 16),
        (9, 16),
    ],
)
def test_get_upper_box_numpy(n, floatx):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-4)
    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)

    x, y, x_0, u_c, W_u, b_u, _, _, _, _, _ = inputs
    x_, y_, x_0_, u_c_, W_u_, b_u_, _, _, _, _, _ = inputs_

    x_expand = x_ + np.zeros_like(x_)
    n_expand = len(W_u_.shape) - len(x_expand.shape)
    for i in range(n_expand):
        x_expand = np.expand_dims(x_expand, -1)

    upper_pred = np.sum(W_u_ * x_expand, 1) + b_u_
    upper_pred = upper_pred.max(0)

    upper = get_upper(x_0, W_u, b_u, {})

    f_upper = K.function([x_0, W_u, b_u], upper)
    upper_ = f_upper([x_0_, W_u_, b_u_]).max()

    assert_allclose(upper_pred, upper_)
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n, floatx",
    [
        (0, 32),
        (1, 32),
        (2, 32),
        (3, 32),
        (4, 32),
        (5, 32),
        (6, 32),
        (7, 32),
        (8, 32),
        (9, 32),
        (0, 64),
        (1, 64),
        (2, 64),
        (3, 64),
        (4, 64),
        (5, 64),
        (6, 64),
        (7, 64),
        (8, 64),
        (9, 64),
        (0, 16),
        (1, 16),
        (2, 16),
        (3, 16),
        (4, 16),
        (5, 16),
        (6, 16),
        (7, 16),
        (8, 16),
        (9, 16),
    ],
)
def test_get_upper_box(n, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-4)
    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)

    x, y, x_0, u_c, W_u, b_u, _, _, _, _, _ = inputs
    _, _, x_0_, u_c_, W_u_, b_u_, _, _, _, _, _ = inputs_

    upper = get_upper(x_0, W_u, b_u, {})

    f_upper = K.function([x_0, W_u, b_u], upper)

    f_u = K.function(inputs, u_c)

    output = inputs_[1]

    upper_ = f_upper([x_0_, W_u_, b_u_])
    u_c_ = f_u(inputs_)

    assert_almost_equal(
        np.clip(output - upper_, 0, np.inf),
        np.zeros_like(output),
        decimal=6,
        err_msg="upper_<y_ in call {}".format(n),
    )

    assert_almost_equal(
        np.clip(u_c_ - upper_, 0, np.inf),
        np.zeros_like(output),
        decimal=6,
        err_msg="upper_<y_ in call {}".format(n),
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n, floatx",
    [
        (0, 32),
        (1, 32),
        (2, 32),
        (3, 32),
        (4, 32),
        (5, 32),
        (6, 32),
        (7, 32),
        (8, 32),
        (9, 32),
        (0, 64),
        (1, 64),
        (2, 64),
        (3, 64),
        (4, 64),
        (5, 64),
        (6, 64),
        (7, 64),
        (8, 64),
        (9, 64),
        (0, 16),
        (1, 16),
        (2, 16),
        (3, 16),
        (4, 16),
        (5, 16),
        (6, 16),
        (7, 16),
        (8, 16),
        (9, 16),
    ],
)
def test_get_lower_box(n, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-4)
    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)
    x, y, x_0, _, _, _, l_c, W_l, b_l, _, _ = inputs

    lower = get_lower(x_0, W_l, b_l, {})

    f_lower = K.function(inputs, lower)
    f_l = K.function(inputs, l_c)

    output = inputs_[1]

    lower_ = f_lower(inputs_)
    l_c_ = f_l(inputs_)

    assert_almost_equal(
        np.clip(lower_ - output, 0, np.inf),
        np.zeros_like(output),
        decimal=6,
        err_msg="lower_> y_ in call {}".format(n),
    )
    assert_almost_equal(
        np.clip(lower_ - l_c_, 0, np.inf),
        np.zeros_like(output),
        decimal=6,
        err_msg="lower_> y_ in call {}".format(n),
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n, floatx",
    [
        (0, 32),
        (1, 32),
        (2, 32),
        (3, 32),
        (4, 32),
        (5, 32),
        (6, 32),
        (7, 32),
        (8, 32),
        (9, 32),
        (0, 64),
        (1, 64),
        (2, 64),
        (3, 64),
        (4, 64),
        (5, 64),
        (6, 64),
        (7, 64),
        (8, 64),
        (9, 64),
        (0, 16),
        (1, 16),
        (2, 16),
        (3, 16),
        (4, 16),
        (5, 16),
        (6, 16),
        (7, 16),
        (8, 16),
        (9, 16),
    ],
)
def test_get_lower_upper_box(n, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-4)
    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)
    x, y, x_0, _, W_u, b_u, _, W_l, b_l, _, _ = inputs

    lower = get_lower(x_0, W_l, b_l, {})
    upper = get_upper(x_0, W_u, b_u, {})

    f_lower = K.function(inputs, lower)
    f_upper = K.function(inputs, upper)

    lower_ = f_lower(inputs_)
    upper_ = f_upper(inputs_)

    assert_almost_equal(
        np.clip(lower_ - upper_, 0, np.inf),
        np.zeros_like(lower_),
        decimal=6,
        err_msg="lower_> upper_ in call {}".format(n),
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n, mode, floatx",
    [
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 32),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
    ],
)
def test_relu_1D_box(n, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)

    if floatx == 16:
        decimal = 2
    else:
        decimal = 5
    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs  # tensors
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
        h_0,
        g_0,
    ) = inputs_  # numpy values

    if mode == "hybrid":
        x_vec = inputs[2:]
        x_vec_ = inputs_[2:]
        output = relu_(x_vec, dc_decomp=True, mode=mode)
    if mode == "ibp":
        x_vec = [u_c, l_c, h, g]
        x_vec_ = [u_c_0, l_c_0, h_0, g_0]
        output = relu_(x_vec, dc_decomp=True, mode=mode)
    if mode == "forward":
        x_vec = [z, W_u, b_u, W_l, b_l, h, g]
        x_vec_ = [z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0]
        output = relu_(x_vec, dc_decomp=True, mode=mode)

    lower = get_lower(z, W_l, b_l)
    upper = get_upper(z, W_u, b_u)

    f_lower = K.function(inputs, lower)
    f_upper = K.function(inputs, upper)

    lower_ = np.min(f_lower(inputs_))
    upper_ = np.max(f_upper(inputs_))

    f_relu_ = K.function(x_vec, output)
    f_ref = K.function(y, K.relu(y))
    y_ = f_ref(y_0)

    # y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_relu_(inputs_[1:])

    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_relu_(x_vec_)
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_relu_(x_vec_)
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_relu_(x_vec_)

    if upper_ <= 0:
        # check that we find this case !
        if mode in ["forward", "hybrid"]:
            assert_almost_equal(
                w_u_,
                np.zeros_like(w_u_),
                decimal=decimal,
                err_msg="w_u_!=0 but upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                b_u_,
                np.zeros_like(b_u_),
                decimal=decimal,
                err_msg="b_u_!=0 but upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                w_l_,
                np.zeros_like(w_l_),
                decimal=decimal,
                err_msg="w_l_!=0 but upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                b_l_,
                np.zeros_like(b_l_),
                decimal=decimal,
                err_msg="b_l_!=0 but upper_<=0 in call {}".format(n),
            )

        if mode in ["ibp", "hybrid"]:
            assert_almost_equal(
                u_c_,
                np.zeros_like(u_c_),
                decimal=decimal,
                err_msg="u_c_!=0 but upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                l_c_,
                np.zeros_like(l_c_),
                decimal=6,
                err_msg="l_c_!=0 but upper_<=0 in call {}".format(n),
            )

    if lower_ >= 0:
        # check that we find this case !
        if mode in ["forward", "hybrid"]:
            assert_almost_equal(
                w_u_,
                W_u_0,
                decimal=decimal,
                err_msg="w_u_!=W_u but lower_>=0 in call {}".format(n),
            )
            assert_almost_equal(
                b_u_,
                b_u_0,
                decimal=decimal,
                err_msg="b_u_!=b_u but lower_>=0  in call {}".format(n),
            )
            assert_almost_equal(
                w_l_,
                W_l_0,
                decimal=decimal,
                err_msg="w_l_!=W_l but lower_>=0 upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                b_l_,
                b_l_0,
                decimal=decimal,
                err_msg="b_l_!=b_l but lower_>=0 upper_<=0 in call {}".format(n),
            )
        if mode in ["ibp", "hybrid"]:
            assert_almost_equal(
                u_c_,
                u_c_0,
                decimal=decimal,
                err_msg="u_c_!=u_c but lower_>=0  in call {}".format(n),
            )
            assert_almost_equal(
                l_c_,
                l_c_0,
                decimal=decimal,
                err_msg="l_c_!=l_c but lower_>=0  in call {}".format(n),
            )

    if mode != "ibp":
        assert_almost_equal(z_0[:, 0], z_[:, 0], decimal=decimal, err_msg="the lower bound should be unchanged")
        assert_almost_equal(z_0[:, 1], z_[:, 1], decimal=decimal, err_msg="the upper bound should be unchanged")

    if mode == "hybrid":
        assert_output_properties_box(
            x_0,
            y_,
            h_,
            g_,
            z_0[:, 0],
            z_0[:, 1],
            u_c_,
            w_u_,
            b_u_,
            l_c_,
            w_l_,
            b_l_,
            "relu_{}".format(n),
            decimal=decimal,
        )

    if mode == "forward":
        assert_output_properties_box(
            x_0,
            y_,
            h_,
            g_,
            z_0[:, 0],
            z_0[:, 1],
            None,
            w_u_,
            b_u_,
            None,
            w_l_,
            b_l_,
            "relu_{}".format(n),
            decimal=decimal,
        )

    if mode == "ibp":
        assert_output_properties_box(
            x_0,
            y_,
            h_,
            g_,
            z_0[:, 0],
            z_0[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "relu_{}".format(n),
            decimal=decimal,
        )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_add(odd, mode, floatx):

    K.set_floatx("float{}".format(32))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)
    if floatx == 16:
        decimal = 2
    else:
        decimal = 5

    inputs_0 = get_tensor_decomposition_multid_box(odd)
    inputs_1 = get_tensor_decomposition_multid_box(odd)

    inputs_ = get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_0

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0, h_0, g_0 = inputs_0
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1, h_1, g_1 = inputs_1
    # x_, y_, z_, u_c_, W_u_, b_u_, l_c_, W_l_, b_l_, h_, g_ = inputs_

    if mode == "hybrid":
        output = add(inputs_0[2:], inputs_1[2:], dc_decomp=True, mode=mode)
    if mode == "forward":
        output = add(
            [z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0],
            [z_1, W_u_1, b_u_1, W_l_1, b_l_1, h_1, g_1],
            dc_decomp=True,
            mode=mode,
        )
    if mode == "ibp":
        output = add([u_c_0, l_c_0, h_0, g_0], [u_c_1, l_c_1, h_1, g_1], dc_decomp=True, mode=mode)

    # TO FINISH

    f_ref = K.function(inputs_0 + inputs_1, inputs_0[1] + inputs_1[1])
    f_add = K.function(inputs_0 + inputs_1, output)

    # y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_add(inputs_ + inputs_)
    y_ = f_ref(inputs_ + inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_add(inputs_ + inputs_)
        assert_output_properties_box(
            inputs_[0],
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
            "add_multid_{}".format(odd),
            decimal=decimal,
        )

    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_add(inputs_ + inputs_)
        assert_output_properties_box(
            inputs_[0],
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
            "add_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_add(inputs_ + inputs_)
        assert_output_properties_box(
            inputs_[0],
            y_,
            h_,
            g_,
            inputs_[2][:, 0],
            inputs_[2][:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "add_multid_{}".format(odd),
            decimal=decimal,
        )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_minus(odd, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)
    if floatx == 16:
        decimal = 2
    else:
        decimal = 5

    inputs_0 = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0, h_0, g_0 = inputs_0
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    if mode == "hybrid":
        output = minus(inputs_0[2:], dc_decomp=True, mode=mode)
    if mode == "forward":
        output = minus([z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0], dc_decomp=True, mode=mode)
    if mode == "ibp":
        output = minus([u_c_0, l_c_0, h_0, g_0], dc_decomp=True, mode=mode)

    f_ref = K.function(inputs_0, -inputs_0[1])
    f_minus = K.function(inputs_0, output)
    y_ = f_ref(inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_minus(inputs_)
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
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_minus(inputs_)
        assert_output_properties_box(
            x,
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
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_minus(inputs_)
        assert_output_properties_box(
            x,
            y_,
            h_,
            g_,
            inputs_[2][:, 0],
            inputs_[2][:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_substract(odd, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)
    if floatx == 16:
        decimal = 2
    else:
        decimal = 5
    inputs_0 = get_tensor_decomposition_multid_box(odd)
    inputs_1 = get_tensor_decomposition_multid_box(odd)

    inputs_ = get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_0

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0, h_0, g_0 = inputs_0
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1, h_1, g_1 = inputs_1
    # x_, y_, z_, u_c_, W_u_, b_u_, l_c_, W_l_, b_l_, h_, g_ = inputs_

    if mode == "hybrid":
        output = substract(inputs_0[2:], inputs_1[2:], dc_decomp=True, mode=mode)
    if mode == "forward":
        output = substract(
            [z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0],
            [z_1, W_u_1, b_u_1, W_l_1, b_l_1, h_1, g_1],
            dc_decomp=True,
            mode=mode,
        )
    if mode == "ibp":
        output = substract([u_c_0, l_c_0, h_0, g_0], [u_c_1, l_c_1, h_1, g_1], dc_decomp=True, mode=mode)

    # TO FINISH

    f_ref = K.function(inputs_0 + inputs_1, inputs_0[1] - inputs_1[1])
    f_sub = K.function(inputs_0 + inputs_1, output)

    # y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_add(inputs_ + inputs_)
    y_ = f_ref(inputs_ + inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_sub(inputs_ + inputs_)
        assert_output_properties_box(
            inputs_[0],
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
            "add_multid_{}".format(odd),
            decimal=decimal,
        )

    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_sub(inputs_ + inputs_)
        assert_output_properties_box(
            inputs_[0],
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
            "add_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_sub(inputs_ + inputs_)
        assert_output_properties_box(
            inputs_[0],
            y_,
            h_,
            g_,
            inputs_[2][:, 0],
            inputs_[2][:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "add_multid_{}".format(odd),
            decimal=decimal,
        )

    K.set_floatx("float{}".format(floatx))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_maximum(odd, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)
    if floatx == 16:
        decimal = 2
    else:
        decimal = 5
    inputs_0 = get_tensor_decomposition_multid_box(odd)
    inputs_1 = get_tensor_decomposition_multid_box(odd)

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0, h_0, g_0 = inputs_0
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1, h_1, g_1 = inputs_1

    inputs_ = get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    if mode == "hybrid":
        output = maximum(inputs_0[2:], inputs_1[2:], dc_decomp=True, mode=mode)
    if mode == "forward":
        output = maximum(
            [z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0],
            [z_1, W_u_1, b_u_1, W_l_1, b_l_1, h_1, g_1],
            dc_decomp=True,
            mode=mode,
        )
    if mode == "ibp":
        output = maximum([u_c_0, l_c_0, h_0, g_0], [u_c_1, l_c_1, h_1, g_1], dc_decomp=True, mode=mode)

    f_ref = K.function(inputs_0 + inputs_1, K.maximum(inputs_0[1], inputs_1[1]))
    f_maximum = K.function(inputs_0 + inputs_1, output)
    y_ = f_ref(inputs_ + inputs_)

    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_maximum(inputs_ + inputs_)
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
            "maximum_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "forward":
        output_ = f_maximum(inputs_ + inputs_)
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = output_
        assert_output_properties_box(
            x,
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
            "maximum_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "ibp":
        output_ = f_maximum(inputs_ + inputs_)
        u_c_, l_c_, h_, g_ = output_
        z_ = inputs_[2]
        assert_output_properties_box(
            x,
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
            "maximum_multid_{}".format(odd),
            decimal=decimal,
        )

    K.set_floatx("float{}".format(floatx))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_max_(odd, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)
    if floatx == 16:
        decimal = 2
    else:
        decimal = 5

    inputs = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    # x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    x_ = inputs_[0]
    z_ = inputs_[2]

    if mode == "hybrid":
        output = max_(inputs[2:], dc_decomp=True, mode=mode)
    if mode == "forward":
        output = max_([z, W_u, b_u, W_l, b_l, h, g], dc_decomp=True, mode=mode)
    if mode == "ibp":
        output = max_([u_c, l_c, h, g], dc_decomp=True, mode=mode)

    f_ref = K.function(inputs, K.max(inputs[1], -1))
    f_max = K.function(inputs, output)

    y_ = f_ref(inputs_)

    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_max(inputs_)
        assert_output_properties_box(
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
            "max_multid_{}".format(odd),
            decimal=decimal,
        )

    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_max(inputs_)
        assert_output_properties_box(
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
            "max_multid_{}".format(odd),
            decimal=decimal,
        )

    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_max(inputs_)
        assert_output_properties_box(
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
            "max_multid_{}".format(odd),
            decimal=decimal,
        )
    K.set_epsilon(eps)
    K.set_floatx("float{}".format(32))


# DC_DECOMP = FALSE


@pytest.mark.parametrize("odd", [0, 1])
def test_max_nodc(odd):

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)

    output = max_(inputs[2:], dc_decomp=False)
    f_max = K.function(inputs, output)
    assert_allclose(len(f_max(inputs_)), 7)


@pytest.mark.parametrize("odd", [0, 1])
def test_maximum_nodc(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)

    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    output = maximum(inputs_0[2:], inputs_1[2:], dc_decomp=False)

    f_ref = K.function(inputs_0 + inputs_1, K.maximum(inputs_0[1], inputs_1[1]))
    f_maximum = K.function(inputs_0 + inputs_1, output)

    assert_allclose(len(f_maximum(inputs_ + inputs_)), 7)
    f_ref(inputs_ + inputs_)


@pytest.mark.parametrize("odd", [0, 1])
def test_minus_nodc(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)

    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    output = minus(inputs_0[2:], dc_decomp=False)

    f_ref = K.function(inputs_0, -inputs_0[1])
    f_minus = K.function(inputs_0, output)

    assert_allclose(len(f_minus(inputs_)), 7)
    f_ref(inputs_)


@pytest.mark.parametrize("odd", [0, 1])
def test_add_nodc(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    output = add(inputs_0[2:], inputs_1[2:], dc_decomp=False)
    f_ref = K.function(inputs_0 + inputs_1, inputs_0[1] + inputs_1[1])
    f_add = K.function(inputs_0 + inputs_1, output)
    assert_allclose(len(f_add(inputs_ + inputs_)), 7)
    f_ref(inputs_ + inputs_)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_relu_1D_box_nodc(n):

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs

    output = relu_(inputs[2:], dc_decomp=False)
    lower = get_lower(z, W_l, b_l)
    upper = get_upper(z, W_u, b_u)

    f_lower = K.function(inputs, lower)
    f_upper = K.function(inputs, upper)

    np.min(f_lower(inputs_))
    np.max(f_upper(inputs_))

    f_relu_ = K.function(inputs[2:], output)
    assert_allclose(len(f_relu_(inputs_[2:])), 7)
