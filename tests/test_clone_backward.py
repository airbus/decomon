# creating toy network and assess that the decomposition is correct


import keras_core.backend as K

from decomon.core import get_affine, get_ibp
from decomon.models.backward_cloning import convert_backward
from decomon.models.forward_cloning import convert_forward


def test_convert_backward_1D(n, mode, floatx, decimal, helpers):
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)
    dc_decomp = False

    #  tensor inputs
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    input_tensors = helpers.get_input_tensors_for_decomon_convert_from_full_inputs(
        inputs=inputs, mode=mode, dc_decomp=dc_decomp
    )

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)
    inputs_ = helpers.prepare_full_np_inputs_for_convert_model(inputs_, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs_)

    # keras model and output of reference
    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    output_ref_ = ref_nn.predict(input_ref_)

    # decomon conversion
    back_bounds = []
    _, _, _, forward_map = convert_forward(
        ref_nn,
        ibp=ibp,
        affine=affine,
        shared=True,
        input_tensors=input_tensors,
        back_bounds=back_bounds,
    )
    _, outputs, _, _ = convert_backward(
        ref_nn,
        input_tensors=input_tensors,
        ibp=ibp,
        affine=affine,
        forward_map=forward_map,
        final_ibp=ibp,
        final_affine=affine,
        back_bounds=back_bounds,
    )

    # decomon outputs
    f_decomon = K.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_decomon_model_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )
