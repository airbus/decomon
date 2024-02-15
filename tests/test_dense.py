import keras.ops as K
import numpy as np
import pytest
from keras.layers import Dense, Input

from decomon.keras_utils import batch_multid_dot
from decomon.layers.core.dense import DecomonDense


def test_decomon_dense(
    use_bias,
    randomize,
    ibp,
    affine,
    propagation,
    perturbation_domain,
    batchsize,
    keras_symbolic_model_input_fn,
    keras_symbolic_layer_input_fn,
    decomon_symbolic_input_fn,
    keras_model_input_fn,
    keras_layer_input_fn,
    decomon_input_fn,
    equal_bounds,
    helpers,
):
    decimal = 4
    units = 7
    decomon_layer_class = DecomonDense

    # init + build keras layer
    keras_symbolic_model_input = keras_symbolic_model_input_fn()
    keras_symbolic_layer_input = keras_symbolic_layer_input_fn(keras_symbolic_model_input)
    layer = Dense(units=units, use_bias=use_bias)
    layer(keras_symbolic_layer_input)

    if randomize:
        # randomize weights => non-zero biases
        for w in layer.weights:
            w.assign(np.random.random(w.shape))

    # init + build decomon layer
    output_shape = layer.output.shape[1:]
    model_output_shape = output_shape
    model_input_shape = keras_symbolic_model_input.shape[1:]
    decomon_symbolic_inputs = decomon_symbolic_input_fn(output_shape=output_shape)
    decomon_layer = decomon_layer_class(
        layer=layer,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
        model_output_shape=model_output_shape,
        model_input_shape=model_input_shape,
    )
    decomon_layer(decomon_symbolic_inputs)

    # call on actual inputs
    keras_model_input = keras_model_input_fn()
    keras_layer_input = keras_layer_input_fn(keras_model_input)
    decomon_inputs = decomon_input_fn(
        keras_model_input=keras_model_input, keras_layer_input=keras_layer_input, output_shape=output_shape
    )

    keras_output = layer(keras_layer_input)
    decomon_output = decomon_layer(decomon_inputs)

    # check affine representation is ok
    w, b = decomon_layer.get_affine_representation()
    keras_output_2 = batch_multid_dot(keras_layer_input, w, missing_batchsize=(False, True)) + b
    np.testing.assert_almost_equal(
        K.convert_to_numpy(keras_output),
        K.convert_to_numpy(keras_output_2),
        decimal=decimal,
        err_msg="wrong affine representation",
    )

    # check output shapes
    input_shape = [t.shape for t in decomon_inputs]
    output_shape = [t.shape for t in decomon_output]
    expected_output_shape = decomon_layer.compute_output_shape(input_shape)
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_layer(
        decomon_output=decomon_output,
        keras_model_input=keras_model_input,
        keras_layer_input=keras_layer_input,
        keras_model_output=keras_output,
        keras_layer_output=keras_output,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        decimal=decimal,
    )

    # before propagation through linear layer lower == upper => lower == upper after propagation
    if equal_bounds:
        helpers.assert_decomon_output_lower_equal_upper(
            decomon_output, ibp=ibp, affine=affine, propagation=propagation, decimal=decimal
        )
