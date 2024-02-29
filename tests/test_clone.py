import pytest
from keras.layers import Dense, Input
from keras.models import Model
from pytest_cases import fixture, parametrize

from decomon.core import ConvertMethod, Propagation, Slope
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
