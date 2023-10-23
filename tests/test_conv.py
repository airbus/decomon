# Test unit for decomon with Dense layers


import keras.config as keras_config
import numpy as np
import pytest
from keras.layers import Conv2D
from tensorflow.python.keras.backend import _get_available_gpus

from decomon.core import ForwardMode, get_affine, get_ibp
from decomon.layers.convert import to_decomon
from decomon.layers.decomon_layers import DecomonConv2D


def test_Decomon_conv_box(data_format, mode, dc_decomp, floatx, decimal, helpers):
    if data_format == "channels_first" and not len(_get_available_gpus()):
        pytest.skip("data format 'channels first' is possible only in GPU mode")

    odd, m_0, m_1 = 0, 0, 1
    kwargs_layer = dict(filters=10, kernel_size=(3, 3), dtype=keras_config.floatx(), data_format=data_format)

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)

    # decomon layer
    decomon_layer = DecomonConv2D(dc_decomp=dc_decomp, mode=mode, **kwargs_layer)

    # original output  (not computed here)
    output_ref_ = None

    # decomon function
    outputs = decomon_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, outputs)

    # test with several kernel (negative or positive part)
    W_, bias = decomon_layer.get_weights()
    for np_function in [np.maximum, np.minimum]:
        decomon_layer.set_weights([np_function(0.0, W_), bias])
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


def test_Decomon_conv_to_decomon_box(shared, floatx, dc_decomp, helpers):
    data_format = "channels_last"
    odd, m_0, m_1 = 0, 0, 1
    dc_decomp = True
    mode = ForwardMode.HYBRID
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)
    kwargs_layer = dict(filters=10, kernel_size=(3, 3), dtype=keras_config.floatx(), data_format=data_format)

    if floatx == 16:
        decimal = 1
    else:
        decimal = 4

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)

    # keras layer & function & output
    keras_layer = Conv2D(**kwargs_layer)
    output_ref = keras_layer(input_ref)
    f_ref = helpers.function(inputs, output_ref)
    output_ref_ = f_ref(inputs_)

    #  decomon layer & function & output via to_decomon
    input_dim = helpers.get_input_dim_from_full_inputs(inputs)
    decomon_layer = to_decomon(keras_layer, input_dim, dc_decomp=dc_decomp, shared=shared, ibp=ibp, affine=affine)
    outputs = decomon_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, outputs)
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
