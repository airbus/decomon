import keras.ops as K
import numpy as np
import pytest
from keras.layers import Dense, Input

from decomon.keras_utils import batch_multid_dot
from decomon.layers.core.dense import DecomonNaiveDense


@pytest.mark.parametrize("input_shape", [(1,), (3,), (5, 2, 3)], ids=["0d", "1d", "multid"])
def test_decomon_dense(use_bias, ibp, affine, propagation, input_shape, perturbation_domain, batchsize, helpers):
    decimal = 5
    units = 7
    output_shape = input_shape[:-1] + (units,)
    keras_symbolic_input = Input(input_shape)
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

    layer = Dense(units=units)
    layer(keras_symbolic_input)

    decomon_layer = DecomonNaiveDense(
        layer=layer, ibp=ibp, affine=affine, propagation=propagation, perturbation_domain=perturbation_domain
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

    # check affine representation is ok
    w, b = decomon_layer.get_affine_representation()
    keras_output_2 = batch_multid_dot(keras_input, w, missing_batchsize=(False, True))
    np.testing.assert_almost_equal(
        K.convert_to_numpy(keras_output),
        K.convert_to_numpy(keras_output_2),
        decimal=decimal,
        err_msg="wrong affine representation",
    )

    decomon_output = decomon_layer(*decomon_inputs)

    # check ibp and affine bounds well ordered w.r.t. keras output
    helpers.assert_decomon_output_compare_with_keras_input_output_single_layer(
        decomon_output=decomon_output, keras_output=keras_output, keras_input=keras_input
    )

    # before propagation through linear layer lower == upper => lower == upper after propagation
    helpers.assert_decomon_output_lower_equal_upper(decomon_output, decimal=decimal)
