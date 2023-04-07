# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.keras.backend as K
from numpy.testing import assert_allclose, assert_almost_equal

from decomon.layers.core import ForwardMode
from decomon.layers.utils import add, get_lower, get_upper, max_, maximum, minus, relu_
from decomon.utils import subtract


def test_get_upper_multi_box(odd, floatx, helpers):

    inputs = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_ = helpers.get_standard_values_multid_box(odd)

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


def test_get_upper_box_numpy(n, floatx, helpers):
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)

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


def test_get_upper_box(n, floatx, helpers):
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)

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


def test_get_lower_box(n, floatx, helpers):
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)
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


def test_get_lower_upper_box(n, floatx, helpers):
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)
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


def test_relu_1D_box(n, mode, floatx, decimal, helpers):
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)

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

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        x_vec = inputs[2:]
        x_vec_ = inputs_[2:]
        output = relu_(x_vec, dc_decomp=True, mode=mode)
    elif mode == ForwardMode.IBP:
        x_vec = [u_c, l_c, h, g]
        x_vec_ = [u_c_0, l_c_0, h_0, g_0]
        output = relu_(x_vec, dc_decomp=True, mode=mode)
    elif mode == ForwardMode.AFFINE:
        x_vec = [z, W_u, b_u, W_l, b_l, h, g]
        x_vec_ = [z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0]
        output = relu_(x_vec, dc_decomp=True, mode=mode)
    else:
        raise ValueError("Unknown mode.")

    lower = get_lower(z, W_l, b_l)
    upper = get_upper(z, W_u, b_u)

    f_lower = K.function(inputs, lower)
    f_upper = K.function(inputs, upper)

    lower_ = np.min(f_lower(inputs_))
    upper_ = np.max(f_upper(inputs_))

    f_relu_ = K.function(x_vec, output)
    f_ref = K.function(y, K.relu(y))
    y_ = f_ref(y_0)

    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_relu_(x_vec_)
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_relu_(x_vec_)
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_relu_(x_vec_)
    else:
        raise ValueError("Unknown mode.")

    if upper_ <= 0:
        # check that we find this case !
        if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
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

        if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
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
        if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
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
        if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
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

    if mode != ForwardMode.IBP:
        assert_almost_equal(z_0[:, 0], z_[:, 0], decimal=decimal, err_msg="the lower bound should be unchanged")
        assert_almost_equal(z_0[:, 1], z_[:, 1], decimal=decimal, err_msg="the upper bound should be unchanged")

    if mode == ForwardMode.HYBRID:
        helpers.assert_output_properties_box(
            x_0, y_, h_, g_, z_0[:, 0], z_0[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=decimal
        )

    elif mode == ForwardMode.AFFINE:
        helpers.assert_output_properties_box(
            x_0, y_, h_, g_, z_0[:, 0], z_0[:, 1], None, w_u_, b_u_, None, w_l_, b_l_, decimal=decimal
        )

    elif mode == ForwardMode.IBP:
        helpers.assert_output_properties_box(
            x_0, y_, h_, g_, z_0[:, 0], z_0[:, 1], u_c_, None, None, l_c_, None, None, decimal=decimal
        )
    else:
        raise ValueError("Unknown mode.")


def test_add(odd, mode, floatx, decimal, helpers):
    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd)

    inputs_ = helpers.get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_0

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0, h_0, g_0 = inputs_0
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1, h_1, g_1 = inputs_1
    # x_, y_, z_, u_c_, W_u_, b_u_, l_c_, W_l_, b_l_, h_, g_ = inputs_

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = add(inputs_0[2:], inputs_1[2:], dc_decomp=True, mode=mode)
    elif mode == ForwardMode.AFFINE:
        output = add(
            [z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0],
            [z_1, W_u_1, b_u_1, W_l_1, b_l_1, h_1, g_1],
            dc_decomp=True,
            mode=mode,
        )
    elif mode == ForwardMode.IBP:
        output = add([u_c_0, l_c_0, h_0, g_0], [u_c_1, l_c_1, h_1, g_1], dc_decomp=True, mode=mode)
    else:
        raise ValueError("Unknown mode.")

    # TO FINISH

    f_ref = K.function(inputs_0 + inputs_1, inputs_0[1] + inputs_1[1])
    f_add = K.function(inputs_0 + inputs_1, output)

    y_ = f_ref(inputs_ + inputs_)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_add(inputs_ + inputs_)
        helpers.assert_output_properties_box(
            inputs_[0], y_, h_, g_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=decimal
        )

    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_add(inputs_ + inputs_)
        helpers.assert_output_properties_box(
            inputs_[0], y_, h_, g_, z_[:, 0], z_[:, 1], None, w_u_, b_u_, None, w_l_, b_l_, decimal=decimal
        )
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_add(inputs_ + inputs_)
        helpers.assert_output_properties_box(
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
            decimal=decimal,
        )
    else:
        raise ValueError("Unknown mode.")


def test_minus(odd, mode, floatx, decimal, helpers):
    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_ = helpers.get_standard_values_multid_box(odd)

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0, h_0, g_0 = inputs_0
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = minus(inputs_0[2:], dc_decomp=True, mode=mode)
    elif mode == ForwardMode.AFFINE:
        output = minus([z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0], dc_decomp=True, mode=mode)
    elif mode == ForwardMode.IBP:
        output = minus([u_c_0, l_c_0, h_0, g_0], dc_decomp=True, mode=mode)
    else:
        raise ValueError("Unknown mode.")

    f_ref = K.function(inputs_0, -inputs_0[1])
    f_minus = K.function(inputs_0, output)
    y_ = f_ref(inputs_)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_minus(inputs_)
        helpers.assert_output_properties_box(
            x, y_, h_, g_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=decimal
        )
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_minus(inputs_)
        helpers.assert_output_properties_box(
            x, y_, h_, g_, z_[:, 0], z_[:, 1], None, w_u_, b_u_, None, w_l_, b_l_, decimal=decimal
        )
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_minus(inputs_)
        helpers.assert_output_properties_box(
            x, y_, h_, g_, inputs_[2][:, 0], inputs_[2][:, 1], u_c_, None, None, l_c_, None, None, decimal=decimal
        )
    else:
        raise ValueError("Unknown mode.")


def test_subtract(odd, mode, floatx, decimal, helpers):
    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd)

    inputs_ = helpers.get_standard_values_multid_box(odd)

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0, h_0, g_0 = inputs_0
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1, h_1, g_1 = inputs_1

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = subtract(inputs_0[2:], inputs_1[2:], dc_decomp=True, mode=mode)
    elif mode == ForwardMode.AFFINE:
        output = subtract(
            [z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0],
            [z_1, W_u_1, b_u_1, W_l_1, b_l_1, h_1, g_1],
            dc_decomp=True,
            mode=mode,
        )
    elif mode == ForwardMode.IBP:
        output = subtract([u_c_0, l_c_0, h_0, g_0], [u_c_1, l_c_1, h_1, g_1], dc_decomp=True, mode=mode)
    else:
        raise ValueError("Unknown mode.")

    # TO FINISH

    f_ref = K.function(inputs_0 + inputs_1, inputs_0[1] - inputs_1[1])
    f_sub = K.function(inputs_0 + inputs_1, output)

    y_ = f_ref(inputs_ + inputs_)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_sub(inputs_ + inputs_)
        helpers.assert_output_properties_box(
            inputs_[0], y_, h_, g_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=decimal
        )

    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_sub(inputs_ + inputs_)
        helpers.assert_output_properties_box(
            inputs_[0], y_, h_, g_, z_[:, 0], z_[:, 1], None, w_u_, b_u_, None, w_l_, b_l_, decimal=decimal
        )
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_sub(inputs_ + inputs_)
        helpers.assert_output_properties_box(
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
            decimal=decimal,
        )
    else:
        raise ValueError("Unknown mode.")


def test_maximum(odd, mode, floatx, decimal, helpers):
    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd)

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0, h_0, g_0 = inputs_0
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1, h_1, g_1 = inputs_1

    inputs_ = helpers.get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = maximum(inputs_0[2:], inputs_1[2:], dc_decomp=True, mode=mode)
    elif mode == ForwardMode.AFFINE:
        output = maximum(
            [z_0, W_u_0, b_u_0, W_l_0, b_l_0, h_0, g_0],
            [z_1, W_u_1, b_u_1, W_l_1, b_l_1, h_1, g_1],
            dc_decomp=True,
            mode=mode,
        )
    elif mode == ForwardMode.IBP:
        output = maximum([u_c_0, l_c_0, h_0, g_0], [u_c_1, l_c_1, h_1, g_1], dc_decomp=True, mode=mode)
    else:
        raise ValueError("Unknown mode.")

    f_ref = K.function(inputs_0 + inputs_1, K.maximum(inputs_0[1], inputs_1[1]))
    f_maximum = K.function(inputs_0 + inputs_1, output)
    y_ = f_ref(inputs_ + inputs_)

    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_maximum(inputs_ + inputs_)
        helpers.assert_output_properties_box(
            x, y_, h_, g_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=decimal
        )
    elif mode == ForwardMode.AFFINE:
        output_ = f_maximum(inputs_ + inputs_)
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = output_
        helpers.assert_output_properties_box(
            x, y_, h_, g_, z_[:, 0], z_[:, 1], None, w_u_, b_u_, None, w_l_, b_l_, decimal=decimal
        )
    elif mode == ForwardMode.IBP:
        output_ = f_maximum(inputs_ + inputs_)
        u_c_, l_c_, h_, g_ = output_
        z_ = inputs_[2]
        helpers.assert_output_properties_box(
            x, y_, h_, g_, z_[:, 0], z_[:, 1], u_c_, None, None, l_c_, None, None, decimal=decimal
        )
    else:
        raise ValueError("Unknown mode.")


def test_max_(odd, mode, floatx, decimal, helpers):

    inputs = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_ = helpers.get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    # x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    x_ = inputs_[0]
    z_ = inputs_[2]

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = max_(inputs[2:], dc_decomp=True, mode=mode)
    elif mode == ForwardMode.AFFINE:
        output = max_([z, W_u, b_u, W_l, b_l, h, g], dc_decomp=True, mode=mode)
    elif mode == ForwardMode.IBP:
        output = max_([u_c, l_c, h, g], dc_decomp=True, mode=mode)
    else:
        raise ValueError("Unknown mode.")

    f_ref = K.function(inputs, K.max(inputs[1], -1))
    f_max = K.function(inputs, output)

    y_ = f_ref(inputs_)

    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_max(inputs_)
        helpers.assert_output_properties_box(
            x_, y_, h_, g_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=decimal
        )

    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_max(inputs_)
        helpers.assert_output_properties_box(
            x_, y_, h_, g_, z_[:, 0], z_[:, 1], None, w_u_, b_u_, None, w_l_, b_l_, decimal=decimal
        )

    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_max(inputs_)
        helpers.assert_output_properties_box(
            x_, y_, h_, g_, z_[:, 0], z_[:, 1], u_c_, None, None, l_c_, None, None, decimal=decimal
        )
    else:
        raise ValueError("Unknown mode.")


# DC_DECOMP = FALSE
def test_max_nodc(odd, helpers):

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)

    output = max_(inputs[2:], dc_decomp=False)
    f_max = K.function(inputs, output)
    assert_allclose(len(f_max(inputs_)), 7)


def test_maximum_nodc(odd, helpers):

    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)

    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    output = maximum(inputs_0[2:], inputs_1[2:], dc_decomp=False)

    f_ref = K.function(inputs_0 + inputs_1, K.maximum(inputs_0[1], inputs_1[1]))
    f_maximum = K.function(inputs_0 + inputs_1, output)

    assert_allclose(len(f_maximum(inputs_ + inputs_)), 7)
    f_ref(inputs_ + inputs_)


def test_minus_nodc(odd, helpers):

    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)

    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    output = minus(inputs_0[2:], dc_decomp=False)

    f_ref = K.function(inputs_0, -inputs_0[1])
    f_minus = K.function(inputs_0, output)

    assert_allclose(len(f_minus(inputs_)), 7)
    f_ref(inputs_)


def test_add_nodc(odd, helpers):

    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    output = add(inputs_0[2:], inputs_1[2:], dc_decomp=False)
    f_ref = K.function(inputs_0 + inputs_1, inputs_0[1] + inputs_1[1])
    f_add = K.function(inputs_0 + inputs_1, output)
    assert_allclose(len(f_add(inputs_ + inputs_)), 7)
    f_ref(inputs_ + inputs_)


def test_relu_1D_box_nodc(n, helpers):

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
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
