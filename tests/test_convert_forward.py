import pytest
from keras.models import Model
from pytest_cases import fixture, parametrize

from decomon.core import Propagation, Slope
from decomon.layers.activations.activation import DecomonBaseActivation
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


def test_convert_forward(
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
    if propagation == Propagation.BACKWARD:
        pytest.skip("backward propagation meaningless for convert_forward()")

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

    # decomon conversion
    decomon_symbolic_output, output_map, layer_map = convert_forward(
        keras_model,
        ibp=ibp,
        affine=affine,
        perturbation_domain_input=decomon_symbolic_input,
        perturbation_domain=perturbation_domain,
        slope=slope,
    )

    # check output map
    def check_maps(output_map, layer_map, keras_model):
        for layer in keras_model.layers:
            for node in layer._inbound_nodes:
                if isinstance(layer, Model):
                    # submodel, iterate on its layers
                    check_maps(output_map, layer_map, layer)
                else:
                    assert id(node) in output_map
                    if len(node.parent_nodes) > 0:  # not an input node
                        assert id(node) in layer_map
                        decomon_layer = layer_map[id(node)]
                        assert decomon_layer.layer is layer
                        if isinstance(decomon_layer, DecomonBaseActivation):
                            assert decomon_layer.slope == slope

    check_maps(output_map, layer_map, keras_model)

    #  decomon outputs
    decomon_model = Model(inputs=decomon_symbolic_input, outputs=decomon_symbolic_output)
    decomon_output = decomon_model(decomon_input)

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_model(
        decomon_output=decomon_output,
        keras_input=keras_input,
        keras_output=keras_output,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        decimal=decimal,
    )
