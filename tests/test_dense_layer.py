# Test unit for decomon with Dense layers


import keras_core.ops as K
import numpy as np
import pytest
from keras_core.layers import Dense
from numpy.testing import assert_almost_equal

from decomon.core import ForwardMode, get_affine, get_ibp
from decomon.layers.convert import to_decomon
from decomon.layers.decomon_layers import DecomonDense
from decomon.models.utils import split_activation


def test_DecomonDense_1D_box(n, mode, shared, floatx, decimal, helpers):
    dc_decomp = True
    kwargs_layer = dict(units=1, use_bias=True, dtype=K.floatx())

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # keras layer & function
    keras_layer = Dense(**kwargs_layer)
    output_ref = keras_layer(input_ref)
    f_ref = K.function(inputs, output_ref)

    # decomon layer & function
    decomon_layer = DecomonDense(dc_decomp=dc_decomp, shared=shared, mode=mode, **kwargs_layer)
    decomon_layer.share_weights(keras_layer)
    outputs = decomon_layer(inputs_for_mode)
    f_decomon = K.function(inputs, outputs)

    # test with several kernel values
    kernel_coeffs = [2, -3]
    W_, bias = decomon_layer.get_weights()

    for kernel_coeff in kernel_coeffs:
        # set weights
        if not shared:
            decomon_layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
        keras_layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

        # keras & decomon outputs
        output_ref_ = f_ref(inputs_)
        outputs_ = f_decomon(inputs_)

        # check bounds consistency
        helpers.assert_decomon_layer_output_properties_box(
            full_inputs=inputs_,
            output_ref=output_ref_,
            outputs_for_mode=outputs_,
            dc_decomp=dc_decomp,
            decimal=decimal,
            mode=mode,
        )


def test_DecomonDense_1D_box_nodc(n, helpers):
    dc_decomp = False
    shared = False
    mode = ForwardMode.HYBRID
    decimal = 5
    kwargs_layer = dict(units=1, use_bias=True, dtype=K.floatx())

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # keras layer & function
    keras_layer = Dense(**kwargs_layer)
    output_ref = keras_layer(input_ref)
    f_ref = K.function(inputs, output_ref)

    #  decomon layer & function
    decomon_layer = DecomonDense(dc_decomp=dc_decomp, shared=shared, mode=mode, **kwargs_layer)
    decomon_layer.share_weights(keras_layer)
    outputs = decomon_layer(inputs_for_mode)
    f_decomon = K.function(inputs, outputs)

    #  test with several kernel values
    kernel_coeffs = [2, -3]
    W_, bias = decomon_layer.get_weights()

    for kernel_coeff in kernel_coeffs:
        # set weights
        if not shared:
            decomon_layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
        keras_layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

        # keras & decomon outputs
        output_ref_ = f_ref(inputs_)
        outputs_ = f_decomon(inputs_)

        # check bounds consistency
        helpers.assert_decomon_layer_output_properties_box_linear(
            full_inputs=inputs_,
            output_ref=output_ref_,
            outputs_for_mode=outputs_,
            dc_decomp=dc_decomp,
            decimal=decimal,
            mode=mode,
        )


def test_DecomonDense_multiD_box(odd, mode, dc_decomp, helpers):
    shared = False
    kwargs_layer = dict(units=1, use_bias=True, dtype=K.floatx())
    decimal = 5

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    # keras layer & function
    keras_layer = Dense(**kwargs_layer)
    output_ref = keras_layer(input_ref)
    f_ref = K.function(inputs, output_ref)

    # decomon layer & function
    decomon_layer = DecomonDense(dc_decomp=dc_decomp, shared=shared, mode=mode, **kwargs_layer)
    decomon_layer.share_weights(keras_layer)
    outputs = decomon_layer(inputs_for_mode)
    f_decomon = K.function(inputs, outputs)

    # test with several kernel values
    kernel_coeffs = [2, -3]
    W_, bias = decomon_layer.get_weights()

    for kernel_coeff in kernel_coeffs:
        # set weights
        if not shared:
            decomon_layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
        keras_layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

        # keras & decomon outputs
        output_ref_ = f_ref(inputs_)
        outputs_ = f_decomon(inputs_)

        # check bounds consistency
        if dc_decomp:
            helpers.assert_decomon_layer_output_properties_box(
                full_inputs=inputs_,
                output_ref=output_ref_,
                outputs_for_mode=outputs_,
                dc_decomp=dc_decomp,
                decimal=decimal,
                mode=mode,
            )
        else:
            helpers.assert_decomon_layer_output_properties_box_linear(
                full_inputs=inputs_,
                output_ref=output_ref_,
                outputs_for_mode=outputs_,
                dc_decomp=dc_decomp,
                decimal=decimal,
                mode=mode,
            )


def test_DecomonDense_1D_to_decomon_box(n, activation, mode, shared, helpers):
    dc_decomp = True
    kwargs_layer = dict(units=1, use_bias=True, dtype=K.floatx())
    decimal = 5
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # keras layer & function & output
    keras_layer = Dense(**kwargs_layer)
    output_ref = keras_layer(input_ref)
    f_ref = K.function(inputs, output_ref)
    output_ref_ = f_ref(inputs_)

    # decomon layer & function & output via to_decomon
    input_dim = helpers.get_input_dim_from_full_inputs(inputs)
    layers_ref = split_activation(keras_layer)
    decomon_layers = []
    for layer in layers_ref:
        decomon_layers.append(to_decomon(layer, input_dim, dc_decomp=dc_decomp, ibp=ibp, affine=affine, shared=shared))
    outputs = decomon_layers[0](inputs_for_mode)
    if len(decomon_layers) > 1:
        outputs = decomon_layers[1](outputs)
    f_decomon = K.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_decomon_layer_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        dc_decomp=dc_decomp,
        decimal=decimal,
        mode=mode,
    )

    # check same weights
    W_, bias = decomon_layers[0].get_weights()
    W_0, b_0 = keras_layer.get_weights()
    assert_almost_equal(W_, W_0, decimal=decimal, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=decimal, err_msg="wrong decomposition")


def test_DecomonDense_multiD_to_decomon_box(odd, activation, mode, dc_decomp, helpers):
    shared = False
    kwargs_layer = dict(units=1, use_bias=True, dtype=K.floatx())
    decimal = 5
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    # keras layer & function & output
    keras_layer = Dense(**kwargs_layer)
    output_ref = keras_layer(input_ref)
    W_0, b_0 = keras_layer.get_weights()
    if odd == 0:
        W_0[0] = 1.037377
        W_0[1] = -0.7575816

        keras_layer.set_weights([W_0, b_0])
    if odd == 1:
        W_0[0] = -0.1657672
        W_0[1] = -0.2613032
        W_0[2] = 0.08437371
        keras_layer.set_weights([W_0, b_0])
    f_ref = K.function(inputs, output_ref)
    output_ref_ = f_ref(inputs_)

    #  decomon layer & function via to_decomon
    input_dim = helpers.get_input_dim_from_full_inputs(inputs)
    layers_ref = split_activation(keras_layer)
    decomon_layers = []
    for layer in layers_ref:
        decomon_layers.append(to_decomon(layer, input_dim, dc_decomp=dc_decomp, ibp=ibp, affine=affine, shared=shared))
    outputs = decomon_layers[0](inputs_for_mode)
    if len(decomon_layers) > 1:
        outputs = decomon_layers[1](outputs)
    f_decomon = K.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    if dc_decomp:
        helpers.assert_decomon_layer_output_properties_box(
            full_inputs=inputs_,
            output_ref=output_ref_,
            outputs_for_mode=outputs_,
            dc_decomp=dc_decomp,
            decimal=decimal,
            mode=mode,
        )
    else:
        helpers.assert_decomon_layer_output_properties_box_linear(
            full_inputs=inputs_,
            output_ref=output_ref_,
            outputs_for_mode=outputs_,
            dc_decomp=dc_decomp,
            decimal=decimal,
            mode=mode,
        )

    # check same weights
    W_, bias = decomon_layers[0].get_weights()
    W_0, b_0 = keras_layer.get_weights()
    assert_almost_equal(W_, W_0, decimal=decimal, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=decimal, err_msg="wrong decomposition")
