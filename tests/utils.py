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

from decomon.layers.utils import get_upper, get_lower, relu_, max_, maximum, add, minus


def test_get_upper_multi_box(odd=0):

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


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_get_upper_box_numpy(n):
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


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_get_upper_box(n):
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


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_get_lower_box(n):
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


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_get_lower_upper_box(n):
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


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_relu_1D_box(n):

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

    output = relu_(inputs[1:], dc_decomp=True)
    lower = get_lower(z, W_l, b_l)
    upper = get_upper(z, W_u, b_u)

    f_lower = K.function(inputs, lower)
    f_upper = K.function(inputs, upper)

    lower_ = np.min(f_lower(inputs_))
    upper_ = np.max(f_upper(inputs_))

    f_relu_ = K.function(inputs[1:], output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_relu_(inputs_[1:])

    if upper_ <= 0:
        # check that we find this case !
        assert_almost_equal(
            w_u_,
            np.zeros_like(w_u_),
            decimal=6,
            err_msg="w_u_!=0 but upper_<=0 in call {}".format(n),
        )
        assert_almost_equal(
            b_u_,
            np.zeros_like(b_u_),
            decimal=6,
            err_msg="b_u_!=0 but upper_<=0 in call {}".format(n),
        )
        assert_almost_equal(
            u_c_,
            np.zeros_like(u_c_),
            decimal=6,
            err_msg="u_c_!=0 but upper_<=0 in call {}".format(n),
        )
        assert_almost_equal(
            l_c_,
            np.zeros_like(l_c_),
            decimal=6,
            err_msg="l_c_!=0 but upper_<=0 in call {}".format(n),
        )
        assert_almost_equal(
            w_l_,
            np.zeros_like(w_l_),
            decimal=6,
            err_msg="w_l_!=0 but upper_<=0 in call {}".format(n),
        )
        assert_almost_equal(
            b_l_,
            np.zeros_like(b_l_),
            decimal=6,
            err_msg="b_l_!=0 but upper_<=0 in call {}".format(n),
        )

    if lower_ >= 0:
        # check that we find this case !
        assert_almost_equal(
            w_u_,
            W_u_0,
            decimal=6,
            err_msg="w_u_!=W_u but lower_>=0 in call {}".format(n),
        )
        assert_almost_equal(
            b_u_,
            b_u_0,
            decimal=6,
            err_msg="b_u_!=b_u but lower_>=0  in call {}".format(n),
        )
        assert_almost_equal(
            u_c_,
            u_c_0,
            decimal=6,
            err_msg="u_c_!=u_c but lower_>=0  in call {}".format(n),
        )
        assert_almost_equal(
            l_c_,
            l_c_0,
            decimal=6,
            err_msg="l_c_!=l_c but lower_>=0  in call {}".format(n),
        )

        assert_almost_equal(
            w_l_,
            W_l_0,
            decimal=6,
            err_msg="w_l_!=W_l but lower_>=0 upper_<=0 in call {}".format(n),
        )
        assert_almost_equal(
            b_l_,
            b_l_0,
            decimal=6,
            err_msg="b_l_!=b_l but lower_>=0 upper_<=0 in call {}".format(n),
        )

    assert_almost_equal(z_0[:, 0], z_[:, 0], decimal=6, err_msg="the lower bound should be unchanged")
    assert_almost_equal(z_0[:, 1], z_[:, 1], decimal=6, err_msg="the upper bound should be unchanged")

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
    )
    y_ = np.maximum(0, y_0)
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
    )


@pytest.mark.parametrize("odd", [0, 1])
def test_add(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd)
    inputs_1 = get_tensor_decomposition_multid_box(odd)

    inputs_ = get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c_, W_l, b_l, h, g = inputs_

    output = add(inputs_0[1:], inputs_1[1:], dc_decomp=True)

    f_ref = K.function(inputs_0 + inputs_1, inputs_0[1] + inputs_1[1])
    f_add = K.function(inputs_0 + inputs_1, output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_add(inputs_ + inputs_)
    y_ = f_ref(inputs_ + inputs_)

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
        "add_multid_{}".format(odd),
    )


@pytest.mark.parametrize("odd", [0, 1])
def test_minus(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_
    output = minus(inputs_0[1:], dc_decomp=True)

    f_ref = K.function(inputs_0, -inputs_0[1])
    f_minus = K.function(inputs_0, output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_minus(inputs_)

    y_ = f_ref(inputs_)

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
    )


@pytest.mark.parametrize("odd", [0, 1])
def test_maximum(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd)
    inputs_1 = get_tensor_decomposition_multid_box(odd)

    inputs_ = get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_
    output = maximum(inputs_0[1:], inputs_1[1:], dc_decomp=True)

    f_ref = K.function(inputs_0 + inputs_1, K.maximum(inputs_0[1], inputs_1[1]))
    f_maximum = K.function(inputs_0 + inputs_1, output)

    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_maximum(inputs_ + inputs_)

    y_ = f_ref(inputs_ + inputs_)

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
    )


@pytest.mark.parametrize("odd", [0, 1])
def test_max_(odd):

    inputs = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output = max_(inputs[1:], dc_decomp=True)
    f_ref = K.function(inputs, K.max(inputs[1], -1))
    f_max = K.function(inputs, output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_max(inputs_)

    y_ = f_ref(inputs_)
    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "max_multid_{}".format(odd),
    )


# DC_DECOMP = FALSE


@pytest.mark.parametrize("odd", [0, 1])
def test_max_nodc(odd):

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)

    output = max_(inputs[1:], dc_decomp=False)
    f_max = K.function(inputs, output)
    assert_allclose(len(f_max(inputs_)), 8)


@pytest.mark.parametrize("odd", [0, 1])
def test_maximum_nodc(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)

    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    output = maximum(inputs_0[1:], inputs_1[1:], dc_decomp=False)

    f_ref = K.function(inputs_0 + inputs_1, K.maximum(inputs_0[1], inputs_1[1]))
    f_maximum = K.function(inputs_0 + inputs_1, output)

    assert_allclose(len(f_maximum(inputs_ + inputs_)), 8)
    f_ref(inputs_ + inputs_)


@pytest.mark.parametrize("odd", [0, 1])
def test_minus_nodc(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)

    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    output = minus(inputs_0[1:], dc_decomp=False)

    f_ref = K.function(inputs_0, -inputs_0[1])
    f_minus = K.function(inputs_0, output)

    assert_allclose(len(f_minus(inputs_)), 8)
    f_ref(inputs_)


@pytest.mark.parametrize("odd", [0, 1])
def test_add_nodc(odd):

    inputs_0 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    output = add(inputs_0[1:], inputs_1[1:], dc_decomp=False)
    f_ref = K.function(inputs_0 + inputs_1, inputs_0[1] + inputs_1[1])
    f_add = K.function(inputs_0 + inputs_1, output)
    assert_allclose(len(f_add(inputs_ + inputs_)), 8)
    f_ref(inputs_ + inputs_)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_relu_1D_box_nodc(n):

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs

    output = relu_(inputs[1:], dc_decomp=False)
    lower = get_lower(z, W_l, b_l)
    upper = get_upper(z, W_u, b_u)

    f_lower = K.function(inputs, lower)
    f_upper = K.function(inputs, upper)

    np.min(f_lower(inputs_))
    np.max(f_upper(inputs_))

    f_relu_ = K.function(inputs[1:], output)
    assert_allclose(len(f_relu_(inputs_[1:])), 8)
