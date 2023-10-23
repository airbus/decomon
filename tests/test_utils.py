# Test unit for decomon with Dense layers


import keras.ops as K
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from decomon.core import BoxDomain, ForwardMode
from decomon.layers.utils import add, max_, maximum, minus, relu_
from decomon.utils import subtract


def test_get_upper_multi_box(odd, floatx, decimal, helpers):
    inputs = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_ = helpers.get_standard_values_multid_box(odd)

    x, y, x_0, u_c, W_u, b_u, _, _, _, _, _ = inputs
    x_, y_, x_0_, u_c_, W_u_, b_u_, _, _, _, _, _ = inputs_

    # compute maximum
    x_min_ = x_0_[:, 0][:, :, None]
    x_max_ = x_0_[:, 1][:, :, None]
    upper_pred = np.sum(np.minimum(W_u_, 0) * x_min_ + np.maximum(W_u_, 0) * x_max_, 1) + b_u_

    upper = BoxDomain().get_upper(x_0, W_u, b_u)

    f_upper = helpers.function([x_0, W_u, b_u], upper)
    upper_ = f_upper([x_0_, W_u_, b_u_])

    assert_allclose(upper_pred, upper_)


def test_get_upper_box_numpy(n, floatx, decimal, helpers):
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

    upper = BoxDomain().get_upper(x_0, W_u, b_u)

    f_upper = helpers.function([x_0, W_u, b_u], upper)
    upper_ = f_upper([x_0_, W_u_, b_u_]).max()

    assert_allclose(upper_pred, upper_)


def test_get_upper_box(n, floatx, decimal, helpers):
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)

    x, y, x_0, u_c, W_u, b_u, _, _, _, _, _ = inputs
    _, _, x_0_, u_c_, W_u_, b_u_, _, _, _, _, _ = inputs_

    upper = BoxDomain().get_upper(x_0, W_u, b_u)

    f_upper = helpers.function([x_0, W_u, b_u], upper)
    f_u = helpers.function(inputs, u_c)

    output = inputs_[1]

    upper_ = f_upper([x_0_, W_u_, b_u_])
    u_c_ = f_u(inputs_)

    assert_almost_equal(
        np.clip(output - upper_, 0, np.inf),
        np.zeros_like(output),
        decimal=decimal,
        err_msg="upper_<y_ in call {}".format(n),
    )

    assert_almost_equal(
        np.clip(u_c_ - upper_, 0, np.inf),
        np.zeros_like(output),
        decimal=decimal,
        err_msg="upper_<y_ in call {}".format(n),
    )


def test_get_lower_box(n, floatx, decimal, helpers):
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)
    x, y, x_0, _, _, _, l_c, W_l, b_l, _, _ = inputs

    lower = BoxDomain().get_lower(x_0, W_l, b_l)

    f_lower = helpers.function(inputs, lower)
    f_l = helpers.function(inputs, l_c)

    input_ref = helpers.get_input_ref_from_full_inputs(inputs_)

    lower_ = f_lower(inputs_)
    l_c_ = f_l(inputs_)

    assert_almost_equal(
        np.clip(lower_ - input_ref, 0, np.inf),
        np.zeros_like(input_ref),
        decimal=decimal,
        err_msg="lower_> y_ in call {}".format(n),
    )
    assert_almost_equal(
        np.clip(lower_ - l_c_, 0, np.inf),
        np.zeros_like(input_ref),
        decimal=decimal,
        err_msg="lower_> y_ in call {}".format(n),
    )


def test_get_lower_upper_box(n, floatx, decimal, helpers):
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)
    x, y, x_0, _, W_u, b_u, _, W_l, b_l, _, _ = inputs

    lower = BoxDomain().get_lower(x_0, W_l, b_l)
    upper = BoxDomain().get_upper(x_0, W_u, b_u)

    f_lower = helpers.function(inputs, lower)
    f_upper = helpers.function(inputs, upper)

    lower_ = f_lower(inputs_)
    upper_ = f_upper(inputs_)

    assert_almost_equal(
        np.clip(lower_ - upper_, 0, np.inf),
        np.zeros_like(lower_),
        decimal=decimal,
        err_msg="lower_> upper_ in call {}".format(n),
    )


def test_relu_1D_box(n, mode, floatx, decimal, helpers):
    dc_decomp = True

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # original output
    f_ref = helpers.function(inputs, K.relu(input_ref))
    output_ref_ = f_ref(inputs_)

    # decomon output
    output = relu_(inputs_for_mode, dc_decomp=dc_decomp, mode=mode)
    f_decomon = helpers.function(inputs, output)
    outputs_ = f_decomon(inputs_)

    # full outputs
    (
        z_output,
        u_c_output,
        w_u_output,
        b_u_output,
        l_c_output,
        w_l_output,
        b_l_output,
        h_output,
        g_output,
    ) = helpers.get_full_outputs_from_outputs_for_mode(
        outputs_for_mode=outputs_,
        full_inputs=inputs_,
        mode=mode,
        dc_decomp=dc_decomp,
    )

    #  check bounds consistency
    helpers.assert_decomon_layer_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )

    # Further checks
    (
        x_tensor,
        y_tensor,
        z_tensor,
        u_c_tensor,
        W_u_tensor,
        b_u_tensor,
        l_c_tensor,
        W_l_tensor,
        b_l_tensor,
        h_tensor,
        g_tensor,
    ) = inputs  # tensors
    (
        x_input,
        y_input,
        z_input,
        u_c_input,
        W_u_input,
        b_u_input,
        l_c_input,
        W_l_input,
        b_l_input,
        h_input,
        g_input,
    ) = inputs_  # numpy values

    # lower and upper bounds from affine coefficients
    lower = BoxDomain().get_lower(z_tensor, W_l_tensor, b_l_tensor)
    upper = BoxDomain().get_upper(z_tensor, W_u_tensor, b_u_tensor)
    f_lower = helpers.function(inputs, lower)
    f_upper = helpers.function(inputs, upper)
    lower_ = np.min(f_lower(inputs_))
    upper_ = np.max(f_upper(inputs_))

    if upper_ <= 0:
        # check that we find this case !
        if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            assert_almost_equal(
                w_u_output,
                np.zeros_like(w_u_output),
                decimal=decimal,
                err_msg="w_u_!=0 but upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                b_u_output,
                np.zeros_like(b_u_output),
                decimal=decimal,
                err_msg="b_u_!=0 but upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                w_l_output,
                np.zeros_like(w_l_output),
                decimal=decimal,
                err_msg="w_l_!=0 but upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                b_l_output,
                np.zeros_like(b_l_output),
                decimal=decimal,
                err_msg="b_l_!=0 but upper_<=0 in call {}".format(n),
            )

        if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            assert_almost_equal(
                u_c_output,
                np.zeros_like(u_c_output),
                decimal=decimal,
                err_msg="u_c_!=0 but upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                l_c_output,
                np.zeros_like(l_c_output),
                decimal=decimal,
                err_msg="l_c_!=0 but upper_<=0 in call {}".format(n),
            )

    if lower_ >= 0:
        # check that we find this case !
        if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            assert_almost_equal(
                w_u_output,
                W_u_input,
                decimal=decimal,
                err_msg="w_u_!=W_u but lower_>=0 in call {}".format(n),
            )
            assert_almost_equal(
                b_u_output,
                b_u_input,
                decimal=decimal,
                err_msg="b_u_!=b_u but lower_>=0  in call {}".format(n),
            )
            assert_almost_equal(
                w_l_output,
                W_l_input,
                decimal=decimal,
                err_msg="w_l_!=W_l but lower_>=0 upper_<=0 in call {}".format(n),
            )
            assert_almost_equal(
                b_l_output,
                b_l_input,
                decimal=decimal,
                err_msg="b_l_!=b_l but lower_>=0 upper_<=0 in call {}".format(n),
            )
        if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            assert_almost_equal(
                u_c_output,
                u_c_input,
                decimal=decimal,
                err_msg="u_c_!=u_c but lower_>=0  in call {}".format(n),
            )
            assert_almost_equal(
                l_c_output,
                l_c_input,
                decimal=decimal,
                err_msg="l_c_!=l_c but lower_>=0  in call {}".format(n),
            )

    if mode != ForwardMode.IBP:
        assert_almost_equal(
            z_input[:, 0], z_output[:, 0], decimal=decimal, err_msg="the lower bound should be unchanged"
        )
        assert_almost_equal(
            z_input[:, 1], z_output[:, 1], decimal=decimal, err_msg="the upper bound should be unchanged"
        )


def add_op(x, y):
    return x + y


def subtract_op(x, y):
    return x - y


def minus_op(x):
    return -x


@pytest.mark.parametrize(
    "decomon_func, tensor_func",
    [
        (add, add_op),
        (subtract, subtract_op),
        (maximum, K.maximum),
    ],
)
def test_func_with_2_inputs(decomon_func, tensor_func, odd, mode, floatx, decimal, helpers):
    dc_decomp = True

    #  tensor inputs
    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_for_mode_0 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_0, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)
    input_ref_0 = helpers.get_input_ref_from_full_inputs(inputs=inputs_0)
    input_ref_1 = helpers.get_input_ref_from_full_inputs(inputs=inputs_1)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd)

    # original output
    f_ref = helpers.function(inputs_0 + inputs_1, tensor_func(input_ref_0, input_ref_1))
    output_ref_ = f_ref(inputs_ + inputs_)

    # decomon output
    output = decomon_func(inputs_for_mode_0, inputs_for_mode_1, dc_decomp=dc_decomp, mode=mode)
    f_decomon = helpers.function(inputs_0 + inputs_1, output)
    outputs_ = f_decomon(inputs_ + inputs_)

    #  check bounds consistency
    helpers.assert_decomon_layer_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )


@pytest.mark.parametrize(
    "decomon_func, tensor_func, tensor_func_kwargs",
    [
        (minus, minus_op, None),
        (max_, K.max, {"axis": -1}),
    ],
)
def test_func_with_1_input(decomon_func, tensor_func, tensor_func_kwargs, odd, mode, floatx, decimal, helpers):
    dc_decomp = True
    if tensor_func_kwargs is None:
        tensor_func_kwargs = {}

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    # original output
    f_ref = helpers.function(inputs, tensor_func(input_ref, **tensor_func_kwargs))
    output_ref_ = f_ref(inputs_)

    # decomon output
    output = decomon_func(inputs_for_mode, dc_decomp=dc_decomp, mode=mode)
    f_decomon = helpers.function(inputs, output)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_decomon_layer_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )


# DC_DECOMP = FALSE
def test_max_nodc(odd, helpers):
    dc_decomp = False
    mode = ForwardMode.HYBRID

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    output = max_(inputs_for_mode, dc_decomp=dc_decomp, mode=mode)
    f_max = helpers.function(inputs, output)
    assert len(f_max(inputs_)) == 7


def test_maximum_nodc(odd, helpers):
    dc_decomp = False
    mode = ForwardMode.HYBRID

    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode_0 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_0, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)
    input_ref_0 = helpers.get_input_ref_from_full_inputs(inputs=inputs_0)
    input_ref_1 = helpers.get_input_ref_from_full_inputs(inputs=inputs_1)

    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)
    output = maximum(inputs_for_mode_0, inputs_for_mode_1, dc_decomp=dc_decomp, mode=mode)

    f_ref = helpers.function(inputs_0 + inputs_1, K.maximum(input_ref_0, input_ref_1))
    f_maximum = helpers.function(inputs_0 + inputs_1, output)

    assert len(f_maximum(inputs_ + inputs_)) == 7
    f_ref(inputs_ + inputs_)


def test_minus_nodc(odd, helpers):
    dc_decomp = False
    mode = ForwardMode.HYBRID

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)
    output = minus(inputs_for_mode, dc_decomp=dc_decomp, mode=mode)

    f_ref = helpers.function(inputs, -inputs[1])
    f_minus = helpers.function(inputs, output)

    assert len(f_minus(inputs_)) == 7
    f_ref(inputs_)


def test_add_nodc(odd, helpers):
    dc_decomp = False
    mode = ForwardMode.HYBRID

    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode_0 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_0, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)
    input_ref_0 = helpers.get_input_ref_from_full_inputs(inputs=inputs_0)
    input_ref_1 = helpers.get_input_ref_from_full_inputs(inputs=inputs_1)

    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    output = add(inputs_for_mode_0, inputs_for_mode_1, dc_decomp=dc_decomp, mode=mode)
    f_ref = helpers.function(inputs_0 + inputs_1, input_ref_0 + input_ref_1)
    f_add = helpers.function(inputs_0 + inputs_1, output)
    assert len(f_add(inputs_ + inputs_)) == 7
    f_ref(inputs_ + inputs_)


def test_relu_1D_box_nodc(n, helpers):
    dc_decomp = False
    mode = ForwardMode.HYBRID

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs

    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    output = relu_(inputs_for_mode, dc_decomp=dc_decomp, mode=mode)
    lower = BoxDomain().get_lower(z, W_l, b_l)
    upper = BoxDomain().get_upper(z, W_u, b_u)

    f_lower = helpers.function(inputs, lower)
    f_upper = helpers.function(inputs, upper)

    np.min(f_lower(inputs_))
    np.max(f_upper(inputs_))

    f_relu_ = helpers.function(inputs, output)
    assert len(f_relu_(inputs_)) == 7
