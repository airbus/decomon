import pytest
from keras_core.layers import Layer, Reshape
from tensorflow.python.keras.backend import _get_available_gpus

from decomon.backward_layers.convert import to_backward
from decomon.core import ForwardMode, Slope
from decomon.layers.decomon_layers import DecomonActivation, DecomonFlatten
from decomon.layers.decomon_reshape import DecomonReshape


def test_Backward_Activation_1D_box_model(n, activation, mode, floatx, decimal, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    # decomon layer
    decomon_layer = DecomonActivation(activation, dc_decomp=dc_decomp, mode=mode, dtype=K.floatx())

    # get backward layer
    backward_layer = to_backward(decomon_layer)

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


def test_Backward_Activation_1D_box_model_slope(helpers):
    n = 2
    activation = "relu"
    mode = ForwardMode.AFFINE
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)

    #  decomon layer
    decomon_layer = DecomonActivation(activation, dc_decomp=dc_decomp, mode=mode, dtype=K.floatx())

    # to_backward with a given slope => outputs
    outputs_by_slope = {}
    for slope in Slope:
        layer_backward = to_backward(decomon_layer, slope=slope, mode=mode)
        assert layer_backward.slope == slope
        outputs = layer_backward(inputs_for_mode)
        f_decomon = helpers.function(inputs, outputs)
        outputs_by_slope[slope] = f_decomon(inputs_)

    # check results
    # O_Slope != Z_Slope
    same_outputs_O_n_Z = [
        (a == b).all() for a, b in zip(outputs_by_slope[Slope.O_SLOPE], outputs_by_slope[Slope.Z_SLOPE])
    ]
    assert not all(same_outputs_O_n_Z)

    # V_Slope == Z_Slope
    for a, b in zip(outputs_by_slope[Slope.V_SLOPE], outputs_by_slope[Slope.Z_SLOPE]):
        assert (a == b).all()


def test_Backward_Activation_multiD_box(odd, activation, floatx, decimal, mode, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd=odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    #  decomon layer
    decomon_layer = DecomonActivation(activation, dc_decomp=dc_decomp, mode=mode, dtype=K.floatx())

    # get backward layer
    backward_layer = to_backward(decomon_layer)

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


def test_Backward_Flatten_multiD_box(odd, floatx, decimal, mode, data_format, helpers):
    if data_format == "channels_first" and not len(_get_available_gpus()):
        pytest.skip("data format 'channels first' is possible only in GPU mode")

    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd=odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    #  decomon layer
    decomon_layer = DecomonFlatten(data_format, dc_decomp=dc_decomp, mode=mode, dtype=K.floatx())

    # get backward layer
    backward_layer = to_backward(decomon_layer)

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


def test_Backward_Reshape_multiD_box(odd, floatx, decimal, mode, helpers):
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd=odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    #  decomon layer
    decomon_layer = DecomonReshape((-1,), dc_decomp=dc_decomp, mode=mode, dtype=K.floatx())

    # get backward layer
    backward_layer = to_backward(decomon_layer)

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


def test_to_backward_ko():
    class MyLayer(Layer):
        ...

    layer = MyLayer()
    with pytest.raises(NotImplementedError):
        to_backward(layer)


def test_name():
    layer = Reshape((1, 2))
    backward_layer = to_backward(layer)
    backward_layer.name.startswith(f"{layer.name}_")
