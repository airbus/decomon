# Test unit for decomon with Dense layers


import keras_core.ops as K
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from decomon.core import ForwardMode, Slope
from decomon.layers.activations import relu, sigmoid, softmax, softsign, tanh


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
def test_activation_1D_box(n, mode, floatx, decimal, helpers, activation_func, tensor_func):
    # softmax: test only n=0,3
    if activation_func is softmax:
        if n not in {0, 3}:
            pytest.skip("softmax activation only possible for n=0 or 3")

    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # reference output
    f_ref = K.function(inputs, tensor_func(input_ref))
    output_ref_ = f_ref(inputs_)

    # decomon output
    output = activation_func(inputs_for_mode, dc_decomp=dc_decomp, mode=mode)
    f_decomon = K.function(inputs, output)
    outputs_ = f_decomon(inputs_)

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
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # decomon output
    outputs = activation_func(inputs_for_mode, dc_decomp=dc_decomp, mode=mode, slope=slope)
    f_decomon = K.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # full outputs & inputs
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

    # check bounds according to slope
    slope = Slope(slope)
    if n == 0:
        assert_almost_equal(w_u_output, np.zeros(w_u_output.shape))
        assert_almost_equal(b_u_output, np.zeros(b_u_output.shape))
        assert_almost_equal(w_l_output, np.zeros(w_l_output.shape))
        assert_almost_equal(b_l_output, np.zeros(b_l_output.shape))
    elif n == 1:
        assert_almost_equal(w_u_output, np.ones(w_u_output.shape))
        assert_almost_equal(b_u_output, np.zeros(b_u_output.shape))
        assert_almost_equal(w_l_output, W_l_0)
        assert_almost_equal(b_l_output, np.zeros(b_l_output.shape))
    elif n == 2:
        assert_almost_equal(w_u_output, 0.5 * np.ones(w_u_output.shape))
        assert_almost_equal(b_u_output, 0.5 * np.ones(b_u_output.shape))
        if slope == Slope.O_SLOPE:
            assert_almost_equal(w_l_output, W_l_0)
            assert_almost_equal(b_l_output, np.zeros(b_l_output.shape))
        elif slope == Slope.Z_SLOPE:
            assert_almost_equal(w_l_output, np.zeros(w_l_output.shape))
            assert_almost_equal(b_l_output, np.zeros(b_l_output.shape))
        elif slope == Slope.V_SLOPE:
            assert_almost_equal(w_l_output, np.zeros(w_l_output.shape))
            assert_almost_equal(b_l_output, np.zeros(b_l_output.shape))
        elif slope == Slope.S_SLOPE:
            assert_almost_equal(w_l_output, w_u_output)
            assert_almost_equal(b_l_output, np.zeros(b_l_output.shape))
        elif slope == Slope.A_SLOPE:
            assert_almost_equal(b_l_output, np.zeros(b_l_output.shape))
            assert_almost_equal(w_l_output, np.zeros(w_l_output.shape))
