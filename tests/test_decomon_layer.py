import keras.ops as K
import numpy as np
import pytest
from keras.layers import Dense, Input

from decomon.core import BoxDomain, Propagation
from decomon.keras_utils import batch_multid_dot
from decomon.layers.layer import DecomonLayer


def test_decomon_layer_nok_unbuilt_keras_layer():
    layer = Dense(3)
    with pytest.raises(ValueError):
        DecomonLayer(layer=layer)


def test_decomon_layer_nok_ibp_affine():
    layer = Dense(3)
    layer(Input((1,)))
    with pytest.raises(ValueError):
        DecomonLayer(layer=layer, ibp=False, affine=False)


def test_decomon_layer_extra_kwargs():
    layer = Dense(3)
    layer(Input((1,)))
    DecomonLayer(layer=layer, alpha="jet")


class MyLinearDecomonDense1d(DecomonLayer):
    linear = True
    layer: Dense

    def get_affine_representation(self):
        return self.layer.kernel, self.layer.bias


class MyNonLinearDecomonDense1d(DecomonLayer):
    linear = False
    layer: Dense

    def get_affine_bounds(self, lower, upper):
        batchsize = lower.shape[0]
        extended_kernel = K.repeat(self.layer.kernel[None], batchsize, axis=0)
        extended_bias = K.repeat(self.layer.bias[None], batchsize, axis=0)
        return extended_kernel, extended_bias, extended_kernel, extended_bias

    def forward_ibp_propagate(self, lower, upper):
        z_value = K.cast(0.0, dtype=lower.dtype)
        kernel_pos = K.maximum(z_value, self.layer.kernel)
        kernel_neg = K.minimum(z_value, self.layer.kernel)
        u_c = K.dot(upper, kernel_pos) + K.dot(lower, kernel_neg)
        l_c = K.dot(lower, kernel_pos) + K.dot(upper, kernel_neg)
        return l_c, u_c


@pytest.mark.parametrize("singlelayer_model", [False, True])
def test_my_decomon_dense_1d(singlelayer_model, ibp, affine, propagation, helpers):
    # input/output shapes
    batchsize = 10
    layer_input_dim = 2
    layer_output_dim = 7
    perturbation_domain = BoxDomain()
    model_output_shape_if_no_singlelayer_model = (3, 5)
    model_input_dim_if_no_singlelayer_model = 9

    layer_input_shape = (layer_input_dim,)
    layer_output_shape = (layer_output_dim,)

    if singlelayer_model:
        model_input_dim = layer_input_dim
        model_output_shape = layer_output_shape
    else:
        model_input_dim = model_input_dim_if_no_singlelayer_model
        model_output_shape = model_output_shape_if_no_singlelayer_model

    model_input_shape = (model_input_dim,)

    x_shape = perturbation_domain.get_x_input_shape_wo_batchsize(model_input_shape)

    # keras layer
    layer = Dense(units=layer_output_dim)
    layer(Input((layer_input_dim,)))

    # decomon layers
    linear_decomon_layer = MyLinearDecomonDense1d(
        layer=layer, ibp=ibp, affine=affine, propagation=propagation, perturbation_domain=perturbation_domain
    )
    non_linear_decomon_layer = MyNonLinearDecomonDense1d(
        layer=layer, ibp=ibp, affine=affine, propagation=propagation, perturbation_domain=perturbation_domain
    )

    # symbolic inputs
    decomon_inputs = helpers.get_decomon_symbolic_inputs(
        model_input_shape=model_input_shape,
        model_output_shape=model_output_shape,
        layer_input_shape=layer_input_shape,
        layer_output_shape=layer_output_shape,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
    )
    affine_bounds_to_propagate, constant_oracle_bounds, x = decomon_inputs

    # actual (random) tensors + expected output shapes
    x_val = helpers.generate_random_tensor(x.shape[1:], batchsize=batchsize)

    if affine:
        if propagation == Propagation.FORWARD:
            b_out_shape = layer_output_shape
            w_out_shape = model_input_shape + layer_output_shape
        else:
            b_out_shape = model_output_shape
            w_out_shape = layer_input_shape + model_output_shape
        affine_bounds_to_propagate_val = [
            helpers.generate_random_tensor(tensor.shape[1:], batchsize=batchsize)
            for tensor in affine_bounds_to_propagate
        ]
        propagated_affine_bounds_expected_shape = [w_out_shape, b_out_shape, w_out_shape, b_out_shape]
    else:
        propagated_affine_bounds_expected_shape = []
        affine_bounds_to_propagate_val = []

    if ibp:
        constant_oracle_bounds_val = [
            helpers.generate_random_tensor(tensor.shape[1:], batchsize=batchsize) for tensor in constant_oracle_bounds
        ]
        propagated_ibp_bounds_expected_shape = [layer_output_shape, layer_output_shape]
    else:
        propagated_ibp_bounds_expected_shape = []
        constant_oracle_bounds_val = []

    decomon_inputs_val = [affine_bounds_to_propagate_val, constant_oracle_bounds_val, x_val]

    if propagation == Propagation.FORWARD:
        decomon_output_expected_shapes = [
            propagated_affine_bounds_expected_shape,
            propagated_ibp_bounds_expected_shape,
        ]
    else:
        decomon_output_expected_shapes = [propagated_affine_bounds_expected_shape]

    # symbolic call
    linear_decomon_output = linear_decomon_layer(*decomon_inputs)
    non_linear_decomon_output = non_linear_decomon_layer(*decomon_inputs)

    # shapes ok ?
    linear_decomon_output_shape_from_call = [[tensor.shape[1:] for tensor in l] for l in linear_decomon_output]
    assert linear_decomon_output_shape_from_call == decomon_output_expected_shapes
    non_linear_decomon_output_shape_from_call = [[tensor.shape[1:] for tensor in l] for l in non_linear_decomon_output]
    assert non_linear_decomon_output_shape_from_call == decomon_output_expected_shapes

    # actual call
    linear_decomon_output_val = linear_decomon_layer(*decomon_inputs_val)
    non_linear_decomon_output_val = non_linear_decomon_layer(*decomon_inputs_val)

    # shapes ok ?
    linear_decomon_output_shape_from_call = [[tensor.shape[1:] for tensor in l] for l in linear_decomon_output_val]
    assert linear_decomon_output_shape_from_call == decomon_output_expected_shapes
    non_linear_decomon_output_shape_from_call = [[tensor.shape[1:] for tensor in l] for l in linear_decomon_output_val]
    assert non_linear_decomon_output_shape_from_call == decomon_output_expected_shapes

    # same values ?
    helpers.assert_decomon_outputs_equal(linear_decomon_output_val, non_linear_decomon_output_val)

    # inequalities hold ?
    # We only check it in case of a single-layer model => model shapes == layer shapes
    # And with identity affine bounds + constant bounds = x = [keras_input, keras_input]
    if model_input_shape == layer_input_shape and model_output_shape == layer_output_shape:
        # keras input
        keras_input_val = helpers.generate_random_tensor(layer_input_shape, batchsize=batchsize)

        # new decomon inputs values
        decomon_inputs_val = helpers.generate_simple_decomon_layer_inputs_from_keras_input(
            keras_input=keras_input_val,
            layer_output_shape=layer_output_shape,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            perturbation_domain=perturbation_domain,
        )

        # decomon call
        linear_decomon_output_val = linear_decomon_layer(*decomon_inputs_val)
        non_linear_decomon_output_val = non_linear_decomon_layer(*decomon_inputs_val)

        # keras call
        keras_output_val = layer(keras_input_val)

        # comparison
        helpers.assert_decomon_output_compare_with_keras_input_output_single_layer(
            decomon_output=linear_decomon_output_val, keras_output=keras_output_val, keras_input=keras_input_val
        )
        helpers.assert_decomon_outputs_equal(linear_decomon_output_val, non_linear_decomon_output_val)