import keras.ops as K
import numpy as np
import pytest
from keras.layers import Input
from keras.models import Model
from pytest_cases import parametrize

from decomon.core import BoxDomain, ConvertMethod, Slope
from decomon.layers.utils.symbolify import LinkToPerturbationDomainInput
from decomon.models.convert import clone


def test_clone_nok_several_inputs():
    a = Input((1,))
    b = Input((2,))
    model = Model([a, b], a)

    with pytest.raises(ValueError, match="only 1 input"):
        clone(model)


@parametrize(
    "toy_model_name",
    [
        "tutorial",
        "tutorial_linear",
        "tutorial_activation_embedded",
        "add",
        "add_linear",
        "merge_v0",
        "merge_v1",
        "merge_v1_seq",
        "merge_v2",
        # "cnn",  # DecomonConv2D not yet implemented
        "embedded_model_v1",
        "embedded_model_v2",
    ],
)
def test_clone(
    toy_model_name,
    toy_model_fn,
    method,
    perturbation_domain,
    model_keras_symbolic_input,
    model_keras_input,
    model_decomon_input,
    helpers,
):
    # input shape?
    input_shape = model_keras_symbolic_input.shape[1:]

    # skip cnn on 0d or 1d input_shape
    if toy_model_name == "cnn" and len(input_shape) == 1:
        pytest.skip("cnn not possible on 0d or 1d input.")

    slope = Slope.Z_SLOPE
    decimal = 4

    # keras model to convert
    keras_model = toy_model_fn(input_shape=input_shape)

    # conversion
    decomon_model = clone(model=keras_model, slope=slope, perturbation_domain=perturbation_domain, method=method)

    # call on actual outputs
    keras_output = keras_model(model_keras_input)
    decomon_output = decomon_model(model_decomon_input)

    ibp = decomon_model.ibp
    affine = decomon_model.affine

    if method in (ConvertMethod.FORWARD_IBP, ConvertMethod.FORWARD_HYBRID):
        assert ibp
    else:
        assert not ibp

    if method == ConvertMethod.FORWARD_IBP:
        assert not affine
    else:
        assert affine

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_model(
        decomon_output=decomon_output,
        keras_input=model_keras_input,
        keras_output=keras_output,
        decimal=decimal,
        ibp=ibp,
        affine=affine,
    )

    # check that we added a layer to insert batch axis
    if toy_model_name.endswith("_linear") and str(method).lower().startswith("crown"):
        assert isinstance(decomon_model.layers[-1], LinkToPerturbationDomainInput)
    else:
        assert not isinstance(decomon_model.layers[-1], LinkToPerturbationDomainInput)


@parametrize(
    "toy_model_name",
    [
        "tutorial",
    ],
)
def test_clone_final_mode(
    toy_model_name,
    toy_model_fn,
    method,
    final_ibp,
    final_affine,
    perturbation_domain,
    simple_model_keras_symbolic_input,
    simple_model_keras_input,
    simple_model_decomon_input,
    helpers,
):
    # input shape?
    input_shape = simple_model_keras_symbolic_input.shape[1:]

    # skip cnn on 0d or 1d input_shape
    if toy_model_name == "cnn" and len(input_shape) == 1:
        pytest.skip("cnn not possible on 0d or 1d input.")

    slope = Slope.Z_SLOPE
    decimal = 4

    # keras model to convert
    keras_model = toy_model_fn(input_shape=input_shape)

    # conversion
    decomon_model = clone(
        model=keras_model,
        slope=slope,
        perturbation_domain=perturbation_domain,
        method=method,
        final_ibp=final_ibp,
        final_affine=final_affine,
    )

    # call on actual outputs
    keras_output = keras_model(simple_model_keras_input)
    decomon_output = decomon_model(simple_model_decomon_input)

    assert final_ibp == decomon_model.ibp
    assert final_affine == decomon_model.affine

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_model(
        decomon_output=decomon_output,
        keras_input=simple_model_keras_input,
        keras_output=keras_output,
        decimal=decimal,
        ibp=final_ibp,
        affine=final_affine,
    )


@parametrize(
    "toy_model_name",
    [
        "tutorial",
    ],
)
@parametrize("equal_ibp, input_shape", [(False, (5, 6, 2))], ids=["multid"])  # fix some parameters of inputs
def test_clone_w_backwardbounds(
    toy_model_name,
    toy_model_fn,
    method,
    perturbation_domain,
    equal_ibp,
    input_shape,
    simple_model_keras_symbolic_input,
    simple_model_keras_input,
    simple_model_decomon_input,
    helpers,
):
    # input shape?
    input_shape = simple_model_keras_symbolic_input.shape[1:]

    slope = Slope.Z_SLOPE
    decimal = 4

    # keras model to convert: chaining 2 models
    keras_model_1 = toy_model_fn(input_shape=input_shape)
    output_shape_1 = keras_model_1.outputs[0].shape  # only 1 output

    keras_model_2 = toy_model_fn(input_shape=output_shape_1[1:])

    input_tot = keras_model_1.inputs[0]
    output_tot = keras_model_2(keras_model_1(input_tot))
    keras_model_tot = Model(input_tot, output_tot)

    # perturbation domain for 2nd model: computed by foward conversion of first model
    forward_model_1 = clone(
        model=keras_model_1,
        slope=slope,
        perturbation_domain=perturbation_domain,
        method=ConvertMethod.FORWARD_HYBRID,
    )
    decomon_input_1 = simple_model_decomon_input
    decomon_output_1 = forward_model_1(decomon_input_1)
    _, _, _, _, lower_ibp, upper_ibp = decomon_output_1
    decomon_input_2 = K.concatenate([lower_ibp[:, None], upper_ibp[:, None]], axis=1)

    # backward_bounds: crown on 2nd model
    crown_model_2 = clone(
        model=keras_model_2,
        slope=slope,
        perturbation_domain=BoxDomain(),
        method=ConvertMethod.CROWN,
    )
    symbolic_backward_bounds = crown_model_2.outputs
    backward_bounds = crown_model_2(decomon_input_2)

    # conversion of first model with backward_bounds
    decomon_model = clone(
        model=keras_model_1,
        slope=slope,
        perturbation_domain=perturbation_domain,
        method=method,
        backward_bounds=symbolic_backward_bounds,
    )

    # call on actual outputs
    keras_output = keras_model_tot(simple_model_keras_input)
    decomon_output = decomon_model([simple_model_decomon_input] + backward_bounds)

    # check output mode
    ibp = decomon_model.ibp
    affine = decomon_model.affine

    if method in (ConvertMethod.FORWARD_IBP, ConvertMethod.FORWARD_HYBRID):
        assert ibp
    else:
        assert not ibp

    if method == ConvertMethod.FORWARD_IBP:
        assert not affine
    else:
        assert affine

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_model(
        decomon_output=decomon_output,
        keras_input=simple_model_keras_input,
        keras_output=keras_output,
        decimal=decimal,
        ibp=ibp,
        affine=affine,
    )


def test_clone_2outputs(
    method,
    final_ibp,
    final_affine,
    perturbation_domain,
    simple_model_keras_symbolic_input,
    simple_model_keras_input,
    simple_model_decomon_input,
    helpers,
):
    # input shape?
    input_shape = simple_model_keras_symbolic_input.shape[1:]

    slope = Slope.Z_SLOPE
    decimal = 4

    # keras model to convert
    keras_model = helpers.toy_network_2outputs(input_shape=input_shape)

    # conversion
    decomon_model = clone(
        model=keras_model,
        slope=slope,
        perturbation_domain=perturbation_domain,
        method=method,
        final_ibp=final_ibp,
        final_affine=final_affine,
    )

    assert final_ibp == decomon_model.ibp
    assert final_affine == decomon_model.affine

    nb_decomon_output_tensor_per_keras_output = 0
    if decomon_model.ibp:
        nb_decomon_output_tensor_per_keras_output += 2
    if decomon_model.affine:
        nb_decomon_output_tensor_per_keras_output += 4
    assert len(decomon_model.outputs) == len(keras_model.outputs) * nb_decomon_output_tensor_per_keras_output

    # call on actual inputs
    keras_output = keras_model(simple_model_keras_input)
    decomon_output = decomon_model(simple_model_decomon_input)

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    for i in range(len(keras_model.outputs)):
        keras_output_i = keras_output[i]
        decomon_output_i = decomon_output[
            i * nb_decomon_output_tensor_per_keras_output : (i + 1) * nb_decomon_output_tensor_per_keras_output
        ]
        helpers.assert_decomon_output_compare_with_keras_input_output_model(
            decomon_output=decomon_output_i,
            keras_input=simple_model_keras_input,
            keras_output=keras_output_i,
            decimal=decimal,
            ibp=final_ibp,
            affine=final_affine,
        )


@parametrize(
    "toy_model_name",
    [
        "tutorial",
    ],
)
@parametrize("equal_ibp, input_shape", [(False, (5, 6, 2))], ids=["multid"])  # fix some parameters of inputs
def test_clone_w_backwardbounds_2outputs(
    toy_model_name,
    toy_model_fn,
    method,
    perturbation_domain,
    equal_ibp,
    input_shape,
    simple_model_keras_symbolic_input,
    simple_model_keras_input,
    simple_model_decomon_input,
    helpers,
):
    # input shape?
    input_shape = simple_model_keras_symbolic_input.shape[1:]

    slope = Slope.Z_SLOPE
    decimal = 4

    # keras model to convert: chaining 2 models (second one can be different for each output of the first one)
    keras_model_1 = helpers.toy_network_2outputs(input_shape=input_shape)
    keras_models_2 = [toy_model_fn(input_shape=output.shape[1:]) for output in keras_model_1.outputs]

    input_tot = keras_model_1.inputs[0]
    outputs_tmp = keras_model_1(input_tot)
    output_tot = [keras_models_2[i](outputs_tmp[i]) for i in range(len(outputs_tmp))]
    keras_model_tot = Model(input_tot, output_tot)

    # perturbation domain for 2nd model: computed by forward conversion of first model
    forward_model_1 = clone(
        model=keras_model_1,
        slope=slope,
        perturbation_domain=perturbation_domain,
        method=ConvertMethod.FORWARD_HYBRID,
    )
    decomon_input_1 = simple_model_decomon_input
    decomon_output_1 = forward_model_1(decomon_input_1)
    _, _, _, _, lower_ibp_1, upper_ibp_1, _, _, _, _, lower_ibp_2, upper_ibp_2 = decomon_output_1
    decomon_inputs_2 = [
        K.concatenate([lower_ibp_1[:, None], upper_ibp_1[:, None]], axis=1),
        K.concatenate([lower_ibp_2[:, None], upper_ibp_2[:, None]], axis=1),
    ]

    # backward_bounds: crown on 2nd model
    crown_models_2 = [
        clone(
            model=keras_model_2,
            slope=slope,
            perturbation_domain=BoxDomain(),
            method=ConvertMethod.CROWN_FORWARD_IBP,
        )
        for keras_model_2 in keras_models_2
    ]
    symbolic_backward_bounds_flattened = [t for crown_model_2 in crown_models_2 for t in crown_model_2.outputs]
    backward_bounds_flattened = [
        t
        for crown_model_2, decomon_input_2 in zip(crown_models_2, decomon_inputs_2)
        for t in crown_model_2(decomon_input_2)
    ]

    # conversion of first model with backward_bounds
    decomon_model = clone(
        model=keras_model_1,
        slope=slope,
        perturbation_domain=perturbation_domain,
        method=method,
        backward_bounds=symbolic_backward_bounds_flattened,
    )

    # check output mode
    ibp = decomon_model.ibp
    affine = decomon_model.affine

    if method in (ConvertMethod.FORWARD_IBP, ConvertMethod.FORWARD_HYBRID):
        assert ibp
    else:
        assert not ibp

    if method == ConvertMethod.FORWARD_IBP:
        assert not affine
    else:
        assert affine

    # check number of outputs
    nb_decomon_output_tensor_per_keras_output = 0
    if decomon_model.ibp:
        nb_decomon_output_tensor_per_keras_output += 2
    if decomon_model.affine:
        nb_decomon_output_tensor_per_keras_output += 4
    assert len(decomon_model.outputs) == len(keras_model_1.outputs) * nb_decomon_output_tensor_per_keras_output

    # call on actual inputs
    keras_output = keras_model_tot(simple_model_keras_input)
    decomon_output = decomon_model([simple_model_decomon_input] + backward_bounds_flattened)

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    for i in range(len(keras_model_1.outputs)):
        keras_output_i = keras_output[i]
        decomon_output_i = decomon_output[
            i * nb_decomon_output_tensor_per_keras_output : (i + 1) * nb_decomon_output_tensor_per_keras_output
        ]
        helpers.assert_decomon_output_compare_with_keras_input_output_model(
            decomon_output=decomon_output_i,
            keras_input=simple_model_keras_input,
            keras_output=keras_output_i,
            decimal=decimal,
            ibp=ibp,
            affine=affine,
        )
