import keras.ops as K
import numpy as np
import pytest
from keras.layers import Dense, Input

from decomon.keras_utils import batch_multid_dot
from decomon.layers.core.dense import DecomonDense, DecomonNaiveDense


@pytest.mark.parametrize("decomon_layer_class", [DecomonNaiveDense, DecomonDense])
def test_decomon_dense(
    decomon_layer_class,
    use_bias,
    randomize,
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
    units = 7

    keras_symbolic_input = keras_symbolic_input_fn()
    input_shape = keras_symbolic_input.shape[1:]
    output_shape = input_shape[:-1] + (units,)
    model_output_shape = output_shape
    decomon_symbolic_inputs = decomon_symbolic_input_fn(output_shape=output_shape)
    keras_input = keras_input_fn()
    decomon_inputs = decomon_input_fn(keras_input=keras_input, output_shape=output_shape)

    layer = Dense(units=units)
    layer(keras_symbolic_input)

    if randomize:
        # randomize weights => non-zero biases
        for w in layer.weights:
            w.assign(np.random.random(w.shape))

    decomon_layer = decomon_layer_class(
        layer=layer,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
        model_output_shape=model_output_shape,
    )
    decomon_layer(decomon_symbolic_inputs)

    keras_output = layer(keras_input)

    # check affine representation is ok
    if decomon_layer_class == DecomonNaiveDense:
        w, b = decomon_layer.get_affine_representation()
        keras_output_2 = batch_multid_dot(keras_input, w, missing_batchsize=(False, True)) + b
        np.testing.assert_almost_equal(
            K.convert_to_numpy(keras_output),
            K.convert_to_numpy(keras_output_2),
            decimal=decimal,
            err_msg="wrong affine representation",
        )

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
    )

    # before propagation through linear layer lower == upper => lower == upper after propagation
    helpers.assert_decomon_output_lower_equal_upper(
        decomon_output, ibp=ibp, affine=affine, propagation=propagation, decimal=decimal
    )
