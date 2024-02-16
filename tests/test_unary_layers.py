import keras.ops as K
import numpy as np
from keras.layers import Activation, Dense
from pytest_cases import fixture, fixture_ref, parametrize

from decomon.keras_utils import batch_multid_dot
from decomon.layers import DecomonActivation, DecomonDense


@fixture
def keras_dense_kwargs(use_bias):
    return dict(units=7, use_bias=use_bias)


@fixture
def keras_activation_kwargs(activation):
    return dict(activation=activation)


@fixture
def decomon_activation_kwargs(slope):
    return dict(slope=slope)


@parametrize(
    "decomon_layer_class, decomon_layer_kwargs, keras_layer_class, keras_layer_kwargs, is_actually_linear",
    [
        (DecomonDense, {}, Dense, keras_dense_kwargs, None),
        (DecomonActivation, decomon_activation_kwargs, Activation, keras_activation_kwargs, None),
    ],
)
def test_decomon_unary_layer(
    decomon_layer_class,
    decomon_layer_kwargs,
    keras_layer_class,
    keras_layer_kwargs,
    is_actually_linear,
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
    if is_actually_linear is None:
        is_actually_linear = decomon_layer_class.linear

    # init + build keras layer
    keras_symbolic_model_input = keras_symbolic_model_input_fn()
    keras_symbolic_layer_input = keras_symbolic_layer_input_fn(keras_symbolic_model_input)
    layer = keras_layer_class(**keras_layer_kwargs)
    layer(keras_symbolic_layer_input)

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
        **decomon_layer_kwargs,
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
    if decomon_layer.linear:
        w, b = decomon_layer.get_affine_representation()
        diagonal = (False, w.shape == b.shape)
        missing_batchsize = (False, True)
        keras_output_2 = (
            batch_multid_dot(keras_layer_input, w, missing_batchsize=missing_batchsize, diagonal=diagonal) + b
        )
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
    if equal_bounds and is_actually_linear:
        helpers.assert_decomon_output_lower_equal_upper(
            decomon_output, ibp=ibp, affine=affine, propagation=propagation, decimal=decimal
        )
