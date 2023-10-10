import keras_core.ops as K

from decomon.metrics.utils import categorical_cross_entropy


def test_categorical_cross_entropy(odd, mode, floatx, decimal, helpers):
    dc_decomp = False

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=dc_decomp)

    # original output
    f_ref = K.function(inputs, -input_ref + K.log(K.sum(K.exp(input_ref), -1))[:, None])
    output_ref_ = f_ref(inputs_)

    # decomon output
    output = categorical_cross_entropy(inputs_for_mode, dc_decomp=dc_decomp, mode=mode)
    f_entropy = K.function(inputs, output)
    outputs_ = f_entropy(inputs_)

    # check bounds consistency
    helpers.assert_decomon_layer_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        dc_decomp=dc_decomp,
        mode=mode,
        decimal=decimal,
    )
