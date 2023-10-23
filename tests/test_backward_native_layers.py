import keras.config as keras_config
import pytest
from keras.layers import Activation, Flatten, Reshape
from tensorflow.python.keras.backend import _get_available_gpus

from decomon.backward_layers.convert import to_backward


def test_Backward_NativeActivation_1D_box_model(n, activation, mode, floatx, decimal, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # keras layer
    keras_layer = Activation(activation, dtype=keras_config.floatx())

    # get backward layer
    backward_layer = to_backward(keras_layer, mode=mode)

    # backward outputs
    outputs = backward_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_backward_layer_output_properties_box_linear(
        full_inputs=inputs_,
        backward_outputs=outputs_,
        decimal=decimal,
    )


def test_Backward_NativeActivation_multiD_box(odd, activation, floatx, decimal, mode, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd=odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    # keras layer
    keras_layer = Activation(activation, dtype=keras_config.floatx())

    # get backward layer
    backward_layer = to_backward(keras_layer, mode=mode)

    # backward outputs
    outputs = backward_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_backward_layer_output_properties_box_linear(
        full_inputs=inputs_,
        backward_outputs=outputs_,
        decimal=decimal,
    )


def test_Backward_NativeFlatten_multiD_box(odd, floatx, decimal, mode, data_format, helpers):
    if data_format == "channels_first" and not len(_get_available_gpus()):
        pytest.skip("data format 'channels first' is possible only in GPU mode")

    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd=odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    # keras layer
    keras_layer = Flatten(data_format, dtype=keras_config.floatx())

    # get backward layer
    backward_layer = to_backward(keras_layer, mode=mode)

    # backward outputs
    outputs = backward_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_backward_layer_output_properties_box_linear(
        full_inputs=inputs_,
        backward_outputs=outputs_,
        decimal=decimal,
    )


def test_Backward_NativeReshape_multiD_box(odd, floatx, decimal, mode, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd=odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    # keras layer
    keras_layer = Reshape((-1,), dtype=keras_config.floatx())

    # get backward layer
    backward_layer = to_backward(keras_layer, mode=mode)

    # backward outputs
    outputs = backward_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_backward_layer_output_properties_box_linear(
        full_inputs=inputs_,
        backward_outputs=outputs_,
        decimal=decimal,
    )
