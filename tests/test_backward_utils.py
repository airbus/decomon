# Test unit for decomon with Dense layers

import keras_core.config as keras_config
import numpy as np
import pytest
from keras_core.layers import Input

from decomon.backward_layers.utils import (
    backward_add,
    backward_maximum,
    backward_subtract,
)


def add_op(x, y):
    return x + y


def subtract_op(x, y):
    return x - y


@pytest.mark.parametrize(
    "n_0, n_1",
    [
        (0, 3),
        (1, 4),
        (2, 5),
    ],
)
@pytest.mark.parametrize(
    "backward_func, tensor_op",
    [
        (backward_add, add_op),
        (backward_maximum, np.maximum),
        (backward_subtract, subtract_op),
    ],
)
def test_reduce_backward_1D_box(n_0, n_1, backward_func, tensor_op, mode, floatx, decimal, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs_0 = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode_0 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_0, mode=mode, dc_decomp=dc_decomp)
    inputs_1 = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_0_ = helpers.get_standard_values_1d_box(n_0, dc_decomp=dc_decomp)
    inputs_1_ = helpers.get_standard_values_1d_box(n_1, dc_decomp=dc_decomp)
    input_ref_0_ = helpers.get_input_ref_from_full_inputs(inputs=inputs_0_)
    batchsize = input_ref_0_.shape[0]

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    # backward outputs
    w_out = Input((1, 1), dtype=keras_config.floatx())
    b_out = Input((1,), dtype=keras_config.floatx())
    back_bounds_0, back_bounds_1 = backward_func(
        inputs_for_mode_0, inputs_for_mode_1, w_out, b_out, w_out, b_out, mode=mode
    )
    f_add = helpers.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    outputs_ = f_add(inputs_0_ + inputs_1_ + [np.ones((batchsize, 1, 1)), np.zeros((batchsize, 1))])
    # get back separate outputs
    outputs_0_ = outputs_[: len(outputs_) // 2]
    outputs_1_ = outputs_[len(outputs_) // 2 :]

    # reference output and constant lower and upper bounds
    output_ref = tensor_op(y_0, y_1)
    upper_constant_bound = tensor_op(u_c_0, u_c_1)
    lower_constant_bound = tensor_op(l_c_0, l_c_1)

    #  check bounds consistency
    helpers.assert_backward_layer_output_properties_box_linear(
        full_inputs=inputs_0_,
        backward_outputs=outputs_0_,
        decimal=decimal,
        output_ref=output_ref,
        upper_constant_bound=upper_constant_bound,
        lower_constant_bound=lower_constant_bound,
    )

    helpers.assert_backward_layer_output_properties_box_linear(
        full_inputs=inputs_1_,
        backward_outputs=outputs_1_,
        decimal=decimal,
        output_ref=output_ref,
        upper_constant_bound=upper_constant_bound,
        lower_constant_bound=lower_constant_bound,
    )
