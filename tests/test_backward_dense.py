import keras.config as keras_config
from keras.layers import Dense, Input

from decomon.backward_layers.convert import to_backward
from decomon.core import ForwardMode
from decomon.layers.decomon_layers import DecomonDense


def test_Backward_Dense_1D_box(n, use_bias, mode, floatx, decimal, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_dim = helpers.get_input_dim_from_full_inputs(inputs)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # keras layer
    keras_layer = Dense(1, use_bias=use_bias, dtype=keras_config.floatx())
    keras_layer(input_ref)  # init weights

    # get backward layer
    backward_layer = to_backward(keras_layer, mode=mode, input_dim=input_dim)

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


def test_Backward_Dense_multiD_box(odd, floatx, decimal, mode, helpers):
    dc_decomp = False
    use_bias = True

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd=odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_dim = helpers.get_input_dim_from_full_inputs(inputs)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    # keras layer
    keras_layer = Dense(1, use_bias=use_bias, dtype=keras_config.floatx())
    keras_layer(input_ref)  # init weights

    # get backward layer
    backward_layer = to_backward(keras_layer, mode=mode, input_dim=input_dim)

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


def test_Backward_DecomonDense_1D_box(n, helpers):
    dc_decomp = False
    use_bias = True
    mode = ForwardMode.HYBRID
    decimal = 5

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_dim = helpers.get_input_dim_from_full_inputs(inputs)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # decomon layer
    decomon_layer = DecomonDense(1, use_bias=use_bias, dc_decomp=dc_decomp, mode=mode, dtype=keras_config.floatx())
    decomon_layer(inputs_for_mode)  # init weights

    # get backward layer
    backward_layer = to_backward(decomon_layer, input_dim=input_dim)

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

    # try to call the backward layer with previous bounds
    w_out = Input((1, 1))
    b_out = Input((1,))
    previous_bounds = [w_out, b_out, w_out, b_out]
    backward_layer(inputs_for_mode + previous_bounds)
