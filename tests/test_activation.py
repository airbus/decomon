# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.keras.backend as K
from numpy.testing import assert_almost_equal

from decomon.layers.activations import relu, sigmoid, softmax, softsign, tanh
from decomon.layers.core import ForwardMode
from decomon.utils import Slope


@pytest.mark.parametrize(
    "activation_func, tensor_func, decimal",
    [
        (sigmoid, K.sigmoid, 5),
        (tanh, K.tanh, 5),
        (softsign, K.softsign, 4),
        (softmax, K.softmax, 4),
        (relu, K.relu, 4),
    ],
)
def test_activation_1D_box(n, mode, floatx, helpers, activation_func, tensor_func, decimal):
    # softmax: test only n=0,3
    if activation_func is softmax:
        if n not in {0, 3}:
            pytest.skip("softmax activation only possible for n=0 or 3")

    mode = ForwardMode(mode)
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-4)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
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

    if mode == ForwardMode.HYBRID:
        output = activation_func(inputs[2:], dc_decomp=False, mode=mode)
    elif mode == ForwardMode.AFFINE:
        output = activation_func([z, W_u, b_u, W_l, b_l], dc_decomp=False, mode=mode)
    elif mode == ForwardMode.IBP:
        output = activation_func([u_c, l_c], dc_decomp=False, mode=mode)
    else:
        raise ValueError("Unknown mode.")

    f_func = K.function(inputs[2:], output)
    f_ref = K.function(inputs, tensor_func(y))
    y_ = f_ref(inputs_)
    z_ = z_0
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_func(inputs_[2:])
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = f_func(inputs_[2:])
        u_c_, l_c_ = [None] * 2
    elif mode == ForwardMode.IBP:
        u_c_, l_c_ = f_func(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
        x_0, y_, None, None, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=decimal
    )

    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n",
    [
        0,
        1,
        2,
    ],
)
def test_activation_1D_box_slope(n, slope, helpers):
    mode = ForwardMode.AFFINE
    activation_func = relu

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
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

    output = activation_func([z, W_u, b_u, W_l, b_l], dc_decomp=False, mode=mode, slope=slope)
    f_func = K.function(inputs[2:], output)
    z_, w_u_, b_u_, w_l_, b_l_ = f_func(inputs_[2:])

    slope = Slope(slope)
    if n == 0:
        assert_almost_equal(w_u_, np.zeros(w_u_.shape))
        assert_almost_equal(b_u_, np.zeros(b_u_.shape))
        assert_almost_equal(w_l_, np.zeros(w_l_.shape))
        assert_almost_equal(b_l_, np.zeros(b_l_.shape))
    elif n == 1:
        assert_almost_equal(w_u_, np.ones(w_u_.shape))
        assert_almost_equal(b_u_, np.zeros(b_u_.shape))
        assert_almost_equal(w_l_, W_l_0)
        assert_almost_equal(b_l_, np.zeros(b_l_.shape))
    elif n == 2:
        assert_almost_equal(w_u_, 0.5 * np.ones(w_u_.shape))
        assert_almost_equal(b_u_, 0.5 * np.ones(b_u_.shape))
        if slope == Slope.O_SLOPE:
            assert_almost_equal(w_l_, W_l_0)
            assert_almost_equal(b_l_, np.zeros(b_l_.shape))
        elif slope == Slope.Z_SLOPE:
            assert_almost_equal(w_l_, np.zeros(w_l_.shape))
            assert_almost_equal(b_l_, np.zeros(b_l_.shape))
        elif slope == Slope.V_SLOPE:
            assert_almost_equal(w_l_, np.zeros(w_l_.shape))
            assert_almost_equal(b_l_, np.zeros(b_l_.shape))
        elif slope == Slope.S_SLOPE:
            assert_almost_equal(w_l_, w_u_)
            assert_almost_equal(b_l_, np.zeros(b_l_.shape))
        elif slope == Slope.A_SLOPE:
            assert_almost_equal(b_l_, np.zeros(b_l_.shape))
            assert_almost_equal(w_l_, np.zeros(w_l_.shape))
