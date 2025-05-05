from keras.models import Model
from pytest_cases import fixture, parametrize

from decomon.constants import Propagation, Slope
from decomon.models.backward_cloning import convert_backward
from decomon.models.forward_cloning import convert_forward


@fixture
@parametrize("name", ["tutorial", "tutorial_linear", "submodel", "submodel_linear", "add", "add_linear", "add_1layer"])
def keras_model_fn(name, helpers):
    if name == "tutorial":
        return helpers.toy_network_tutorial
    elif name == "tutorial_linear":
        return lambda input_shape, dtype=None: helpers.toy_network_tutorial(
            input_shape=input_shape, dtype=dtype, activation=None
        )
    elif name == "add":
        return helpers.toy_network_add
    elif name == "add_linear":
        return lambda input_shape, dtype=None: helpers.toy_network_add(
            input_shape=input_shape, dtype=dtype, activation=None
        )
    elif name == "add_1layer":
        return helpers.toy_network_add_monolayer
    elif name == "submodel":
        return helpers.toy_network_submodel
    elif name == "submodel_linear":
        return lambda input_shape, dtype=None: helpers.toy_network_submodel(
            input_shape=input_shape, dtype=dtype, activation=None
        )
    else:
        raise ValueError()


def test_convert_backward(
    ibp,
    affine,
    propagation,
    perturbation_domain,
    input_shape,
    keras_model_fn,
    simple_model_decomon_symbolic_input_fn,
    simple_model_keras_input_fn,
    simple_model_decomon_input_fn,
    helpers,
):
    slope = Slope.Z_SLOPE
    decimal = 4

    # keras model to convert
    keras_model = keras_model_fn(input_shape=input_shape)

    # symbolic inputs
    keras_symbolic_input = keras_model.inputs[0]
    decomon_symbolic_input = simple_model_decomon_symbolic_input_fn(keras_symbolic_input)

    # actual inputs
    keras_input = simple_model_keras_input_fn(keras_symbolic_input)
    decomon_input = simple_model_decomon_input_fn(keras_input)

    # keras output
    keras_output = keras_model(keras_input)

    # forward conversion for forward oracle
    if propagation == Propagation.FORWARD:
        _, forward_output_map, forward_layer_map = convert_forward(
            keras_model, ibp=ibp, affine=affine, perturbation_domain_input=decomon_symbolic_input, slope=slope
        )
    else:
        forward_output_map, forward_layer_map = None, None

    # backward conversion
    decomon_symbolic_output = convert_backward(
        keras_model,
        perturbation_domain_input=decomon_symbolic_input,
        perturbation_domain=perturbation_domain,
        slope=slope,
        forward_output_map=forward_output_map,
        forward_layer_map=forward_layer_map,
    )

    #  decomon outputs
    if None in decomon_symbolic_output[0].shape:
        decomon_model = Model(inputs=decomon_symbolic_input, outputs=decomon_symbolic_output)
        decomon_output = decomon_model(decomon_input)
    else:
        # special case for pure linear keras model => bounds not depending on batch, already computed.
        decomon_output = decomon_symbolic_output

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_model(
        decomon_output=decomon_output,
        keras_input=keras_input,
        keras_output=keras_output,
        propagation=Propagation.BACKWARD,
        decimal=decimal,
        ibp=ibp,
        affine=affine,
    )
