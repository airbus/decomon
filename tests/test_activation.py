import keras.ops as K
import numpy as np
import pytest
from keras.layers import Activation, Dense, Input

from decomon.keras_utils import batch_multid_dot
from decomon.layers.activations.activation import DecomonActivation


def test_decomon_activation(
    activation,
    slope,
    ibp,
    affine,
    propagation,
    perturbation_domain,
    batchsize,
    keras_symbolic_input_fn,
    decomon_symbolic_input_fn,
    keras_input_fn,
    decomon_input_fn,
    helpers,
):
    decimal = 5
    decomon_layer_class = DecomonActivation

    # init + build keras layer
    keras_symbolic_input = keras_symbolic_input_fn()
    layer = Activation(activation=activation)
    layer(keras_symbolic_input)

    # init + build decomon layer
    output_shape = layer.output.shape[1:]
    model_output_shape = output_shape
    decomon_symbolic_inputs = decomon_symbolic_input_fn(output_shape=output_shape)
    decomon_layer = decomon_layer_class(
        layer=layer,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
        model_output_shape=model_output_shape,
        slope=slope,
    )
    decomon_layer(decomon_symbolic_inputs)

    # call on actual inputs
    keras_input = keras_input_fn()
    decomon_inputs = decomon_input_fn(keras_input=keras_input, output_shape=output_shape)

    keras_output = layer(keras_input)
    decomon_output = decomon_layer(decomon_inputs)

    # check output shapes
    input_shape = [t.shape for t in decomon_inputs]
    output_shape = [t.shape for t in decomon_output]
    expected_output_shape = decomon_layer.compute_output_shape(input_shape)
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape

    # check ibp and affine bounds well ordered w.r.t. keras output
    helpers.assert_decomon_output_compare_with_keras_input_output_single_layer(
        decomon_output=decomon_output,
        keras_output=keras_output,
        keras_input=keras_input,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        decimal=decimal,
    )
