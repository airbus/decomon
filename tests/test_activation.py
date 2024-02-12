import keras.ops as K
import numpy as np
import pytest
from keras.layers import Activation, Dense, Input

from decomon.keras_utils import batch_multid_dot
from decomon.layers.activations.activation import DecomonActivation


@pytest.mark.parametrize("input_shape", [(1,), (3,), (5, 2, 3)], ids=["0d", "1d", "multid"])
def test_decomon_activation(
    activation,
    slope,
    ibp,
    affine,
    propagation,
    input_shape,
    perturbation_domain,
    batchsize,
    helpers,
):
    decimal = 5
    decomon_layer_class = DecomonActivation

    keras_symbolic_input = Input(input_shape)
    layer = Activation(activation=activation)
    layer(keras_symbolic_input)
    output_shape = layer.output.shape[1:]
    model_output_shape_length = len(output_shape)

    decomon_symbolic_inputs = helpers.get_decomon_symbolic_inputs(
        model_input_shape=input_shape,
        model_output_shape=output_shape,
        layer_input_shape=input_shape,
        layer_output_shape=output_shape,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
    )
    decomon_layer = decomon_layer_class(
        layer=layer,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
        model_output_shape_length=model_output_shape_length,
        slope=slope,
    )
    decomon_layer(*decomon_symbolic_inputs)

    keras_input = helpers.generate_random_tensor(input_shape, batchsize=batchsize)
    decomon_inputs = helpers.generate_simple_decomon_layer_inputs_from_keras_input(
        keras_input=keras_input,
        layer_output_shape=output_shape,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
    )

    keras_output = layer(keras_input)

    decomon_output = decomon_layer(*decomon_inputs)

    # check ibp and affine bounds well ordered w.r.t. keras output
    helpers.assert_decomon_output_compare_with_keras_input_output_single_layer(
        decomon_output=decomon_output, keras_output=keras_output, keras_input=keras_input
    )
