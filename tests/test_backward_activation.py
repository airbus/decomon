import keras_core.backend as K
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from decomon.backward_layers.activations import backward_relu, backward_softsign
from decomon.core import ForwardMode, Slope


@pytest.mark.parametrize(
    "activation_func",
    [
        backward_relu,
        backward_softsign,
    ],
)
def test_activation_backward_1D_box(n, mode, floatx, decimal, activation_func, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # backward outputs
    outputs = activation_func(inputs_for_mode, mode=mode)
    f_func = K.function(inputs, outputs)
    outputs_ = f_func(inputs_)

    # check bounds consistency
    helpers.assert_backward_layer_output_properties_box_linear(
        full_inputs=inputs_,
        backward_outputs=outputs_,
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
def test_activation_backward_1D_box_slope(n, slope, helpers):
    mode = ForwardMode.AFFINE
    activation_func = backward_relu
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)
    x_, y_, z_, u_c_, W_u_, B_u_, l_c_, W_l_, B_l_ = inputs_

    # backward outputs
    outputs = activation_func(inputs_for_mode, mode=mode)
    f_func = K.function(inputs, outputs)
    outputs_ = f_func(inputs_)

    # backward recomposition
    w_u_, b_u_, w_l_, b_l_ = outputs_
    w_u_b = np.sum(np.maximum(w_u_, 0) * W_u_ + np.minimum(w_u_, 0) * W_l_, 1)[:, :, None]
    b_u_b = b_u_ + np.sum(np.maximum(w_u_, 0) * B_u_[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * B_l_[:, :, None], 1)
    w_l_b = np.sum(np.maximum(w_l_, 0) * W_l_ + np.minimum(w_l_, 0) * W_u_, 1)[:, :, None]
    b_l_b = b_l_ + np.sum(np.maximum(w_l_, 0) * B_l_[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * B_u_[:, :, None], 1)

    # check bounds according to slope
    slope = Slope(slope)
    if n == 0:
        assert_almost_equal(w_u_b, np.zeros(w_u_b.shape))
        assert_almost_equal(b_u_b, np.zeros(b_u_b.shape))
        assert_almost_equal(w_l_b, np.zeros(w_l_b.shape))
        assert_almost_equal(b_l_b, np.zeros(b_l_b.shape))
    elif n == 1:
        assert_almost_equal(w_u_b, len(w_u_b) * np.ones(w_u_b.shape))  # * len ??
        assert_almost_equal(b_u_b, np.zeros(b_u_b.shape))
        assert_almost_equal(w_l_b, len(w_l_b) * np.ones(w_l_b.shape))
        assert_almost_equal(b_l_b, np.zeros(b_l_b.shape))
    elif n == 2:
        assert_almost_equal(w_u_b, 0.5 * len(w_u_b) * np.ones(w_u_b.shape))
        assert_almost_equal(b_u_b, 0.5 * np.ones(b_u_b.shape))
        if slope == Slope.O_SLOPE:
            assert_almost_equal(w_l_b, np.zeros(w_l_b.shape))  # ??
            assert_almost_equal(b_l_b, np.zeros(b_l_b.shape))
        elif slope == Slope.Z_SLOPE:
            assert_almost_equal(w_l_b, np.zeros(w_l_b.shape))
            assert_almost_equal(b_l_b, np.zeros(b_l_b.shape))
        elif slope == Slope.V_SLOPE:
            assert_almost_equal(w_l_b, np.zeros(w_l_b.shape))
            assert_almost_equal(b_l_b, np.zeros(b_l_b.shape))
        elif slope == Slope.S_SLOPE:
            assert_almost_equal(w_l_b, np.zeros(w_l_b.shape))  # ??
            assert_almost_equal(b_l_b, np.zeros(b_l_b.shape))
        elif slope == Slope.A_SLOPE:
            assert_almost_equal(b_l_b, np.zeros(b_l_b.shape))
            assert_almost_equal(w_l_b, np.zeros(w_l_b.shape))
