# Test unit for decomon with Dense layers


import numpy as np
import pytest
from keras_core.layers import (
    Add,
    Average,
    Concatenate,
    Maximum,
    Minimum,
    Multiply,
    Subtract,
)

from decomon.core import ForwardMode, get_affine, get_ibp
from decomon.layers.convert import to_decomon
from decomon.layers.decomon_merge_layers import (
    DecomonAdd,
    DecomonAverage,
    DecomonConcatenate,
    DecomonMaximum,
    DecomonMinimum,
    DecomonMultiply,
    DecomonSubtract,
)


def add_op(x, y):
    return x + y


def subtract_op(x, y):
    return x - y


def multiply_op(x, y):
    return x * y


def average_op(x, y):
    return (x + y) / 2.0


def concatenate_op(x, y):
    return np.concatenate([x, y], -1)


@pytest.mark.parametrize(
    "decomon_op_class, tensor_op, decomon_op_kwargs",
    [
        (DecomonAdd, add_op, {}),
        (DecomonSubtract, subtract_op, {}),
        (DecomonAverage, average_op, {}),
        (DecomonMaximum, np.maximum, {}),
        (DecomonMinimum, np.minimum, {}),
        (DecomonConcatenate, concatenate_op, {"axis": -1}),
        (DecomonMultiply, multiply_op, {}),
    ],
)
def test_DecomonOp_1D_box(decomon_op_class, tensor_op, decomon_op_kwargs, n, mode, floatx, decimal, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs_0 = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_1 = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode_0 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_0, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs=inputs_)

    # original output
    output_ref_ = tensor_op(input_ref_, input_ref_)

    # decomon output
    decomon_op = decomon_op_class(dc_decomp=dc_decomp, mode=mode, dtype=K.floatx(), **decomon_op_kwargs)
    output = decomon_op(inputs_for_mode_0 + inputs_for_mode_1)
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
    "decomon_op_class, tensor_op, decomon_op_kwargs",
    [
        (DecomonAdd, add_op, {}),
        (DecomonSubtract, subtract_op, {}),
        (DecomonAverage, average_op, {}),
        (DecomonMaximum, np.maximum, {}),
        (DecomonMinimum, np.minimum, {}),
        (DecomonConcatenate, concatenate_op, {"axis": -1}),
        (DecomonMultiply, multiply_op, {}),
    ],
)
def test_DecomonOp_multiD_box(decomon_op_class, tensor_op, decomon_op_kwargs, odd, mode, floatx, decimal, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode_0 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_0, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs=inputs_)

    # original output
    output_ref_ = tensor_op(input_ref_, input_ref_)

    # decomon output
    decomon_op = decomon_op_class(dc_decomp=dc_decomp, mode=mode, dtype=K.floatx(), **decomon_op_kwargs)
    output = decomon_op(inputs_for_mode_0 + inputs_for_mode_1)
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
    "layer_class, tensor_op, layer_kwargs",
    [
        (Add, add_op, {}),
        (Average, average_op, {}),
        (Subtract, subtract_op, {}),
        (Maximum, np.maximum, {}),
        (Minimum, np.minimum, {}),
        (Concatenate, concatenate_op, {"axis": -1}),
        (Multiply, multiply_op, {}),
    ],
)
def test_Decomon_1D_box_to_decomon(layer_class, tensor_op, layer_kwargs, n, helpers):
    dc_decomp = False
    mode = ForwardMode.HYBRID
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)
    decimal = 5

    #  tensor inputs
    inputs_0 = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_1 = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode_0 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_0, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)
    input_ref_0 = helpers.get_input_ref_from_full_inputs(inputs=inputs_0)
    input_ref_1 = helpers.get_input_ref_from_full_inputs(inputs=inputs_1)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs=inputs_)

    # original output
    output_ref_ = tensor_op(input_ref_, input_ref_)

    # to_decomon
    ref_op = layer_class(dtype=K.floatx(), **layer_kwargs)
    ref_op([input_ref_0, input_ref_1])
    decomon_op = to_decomon(
        ref_op, input_dim=helpers.get_input_dim_from_full_inputs(inputs_0), dc_decomp=dc_decomp, affine=affine, ibp=ibp
    )

    # decomon output
    output = decomon_op(inputs_for_mode_0 + inputs_for_mode_1)
    f_decomon = helpers.function(inputs_0 + inputs_1, output)
    outputs_ = f_decomon(inputs_ + inputs_)

    #  check bounds consistency
    helpers.assert_decomon_layer_output_properties_box_linear(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )


#### to_decomon multiD


@pytest.mark.parametrize(
    "layer_class, tensor_op, layer_kwargs",
    [
        (Add, add_op, {}),
        (Average, average_op, {}),
        (Subtract, subtract_op, {}),
        (Maximum, np.maximum, {}),
        (Minimum, np.minimum, {}),
        (Concatenate, concatenate_op, {"axis": -1}),
        (Multiply, multiply_op, {}),
    ],
)
def test_Decomon_multiD_box_to_decomon(layer_class, tensor_op, layer_kwargs, odd, helpers):
    dc_decomp = False
    mode = ForwardMode.HYBRID
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)
    decimal = 5

    #  tensor inputs
    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode_0 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_0, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)
    input_ref_0 = helpers.get_input_ref_from_full_inputs(inputs=inputs_0)
    input_ref_1 = helpers.get_input_ref_from_full_inputs(inputs=inputs_1)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs=inputs_)

    # original output
    output_ref_ = tensor_op(input_ref_, input_ref_)

    #  to_decomon
    ref_op = layer_class(dtype=K.floatx(), **layer_kwargs)
    ref_op([input_ref_0, input_ref_1])
    decomon_op = to_decomon(
        ref_op, input_dim=helpers.get_input_dim_from_full_inputs(inputs_0), dc_decomp=dc_decomp, affine=affine, ibp=ibp
    )

    # decomon output
    output = decomon_op(inputs_for_mode_0 + inputs_for_mode_1)
    f_decomon = helpers.function(inputs_0 + inputs_1, output)
    outputs_ = f_decomon(inputs_ + inputs_)

    #  check bounds consistency
    helpers.assert_decomon_layer_output_properties_box_linear(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )
