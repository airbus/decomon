import keras_core.backend as K
from keras_core.layers import MaxPooling2D

from decomon.layers.maxpooling import DecomonMaxPooling2D


def test_MaxPooling2D_box(mode, floatx, decimal, helpers):
    odd, m_0, m_1 = 0, 0, 1
    data_format = "channels_last"
    dc_decomp = True
    fast = False
    kwargs_layer = dict(pool_size=(2, 2), strides=(2, 2), padding="valid", dtype=K.floatx())

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)

    # keras & decomon layer
    keras_layer = MaxPooling2D(**kwargs_layer)
    decomon_layer = DecomonMaxPooling2D(dc_decomp=dc_decomp, fast=fast, mode=mode, **kwargs_layer)

    # original output
    output_ref = keras_layer(input_ref)
    f_ref = K.function(inputs, output_ref)
    output_ref_ = f_ref(inputs_)

    # decomon outputs
    outputs = decomon_layer(inputs_for_mode)
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
