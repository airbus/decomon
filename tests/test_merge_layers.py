from typing import TypeVar

import keras.ops as K
import numpy as np
import pytest
from keras.layers import Add

from decomon.keras_utils import batch_multid_dot
from decomon.layers.merging.add import DecomonAdd, DecomonMerge
from decomon.types import Tensor


# Defining non-linear and/or non-diagonal versions of DecomonAdd
class DecomonNonDiagAdd(DecomonMerge):
    layer: Add
    linear = True
    diagonal = False

    def get_affine_representation(self) -> tuple[list[Tensor], Tensor]:
        w = []
        for input_i in self.keras_layer_input:
            diag_shape = input_i.shape[1:]
            full_shape = diag_shape + diag_shape
            flattened_diag_shape = (int(np.prod(diag_shape)),)
            w.append(K.reshape(K.diag(K.ones(flattened_diag_shape)), full_shape))

        b = K.zeros(self.layer.output.shape[1:])

        return w, b


class DecomonNonLinearAdd(DecomonMerge):
    layer: Add
    linear = False
    diagonal = True

    def get_affine_bounds(
        self, lower: list[Tensor], upper: list[Tensor]
    ) -> tuple[list[Tensor], Tensor, list[Tensor], Tensor]:
        w = []
        batchsize = lower[0].shape[0]
        for input_i in self.keras_layer_input:
            diag_shape = input_i.shape[1:]
            w.append(K.repeat(K.ones(diag_shape)[None], batchsize, axis=0))

        b = K.zeros_like(lower[0])

        return w, b, w, b

    def forward_ibp_propagate(self, lower: list[Tensor], upper: list[Tensor]) -> tuple[Tensor, Tensor]:
        return sum(lower), sum(upper)


class DecomonNonLinearNonDiagAdd(DecomonMerge):
    layer: Add
    linear = False
    diagonal = False

    def get_affine_bounds(
        self, lower: list[Tensor], upper: list[Tensor]
    ) -> tuple[list[Tensor], Tensor, list[Tensor], Tensor]:
        w = []
        batchsize = lower[0].shape[0]
        for input_i in self.keras_layer_input:
            diag_shape = input_i.shape[1:]
            full_shape = diag_shape + diag_shape
            flattened_diag_shape = (int(np.prod(diag_shape)),)
            w.append(K.repeat(K.reshape(K.diag(K.ones(flattened_diag_shape)), full_shape)[None], batchsize, axis=0))

        b = K.zeros_like(lower[0])

        return w, b, w, b

    def forward_ibp_propagate(self, lower: list[Tensor], upper: list[Tensor]) -> tuple[Tensor, Tensor]:
        return sum(lower), sum(upper)


T = TypeVar("T")


def double_input(input: T) -> list[T]:
    return [input] * 2


@pytest.mark.parametrize(
    "decomon_layer_class, decomon_layer_kwargs, keras_layer_class, keras_layer_kwargs, is_actually_linear",
    [
        (DecomonAdd, {}, Add, {}, True),
        (DecomonNonDiagAdd, {}, Add, {}, True),
        (DecomonNonLinearAdd, {}, Add, {}, True),
        (DecomonNonLinearNonDiagAdd, {}, Add, {}, True),
    ],
)
def test_decomon_merge(
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
    keras_symbolic_layer_input_0 = keras_symbolic_layer_input_fn(keras_symbolic_model_input)
    # we merge twice the same input
    keras_symbolic_layer_input = double_input(keras_symbolic_layer_input_0)
    layer = keras_layer_class(**keras_layer_kwargs)
    layer(keras_symbolic_layer_input)

    # init + build decomon layer
    output_shape = layer.output.shape[1:]
    model_output_shape = output_shape
    model_input_shape = keras_symbolic_model_input.shape[1:]
    decomon_symbolic_inputs_0 = decomon_symbolic_input_fn(output_shape=output_shape)
    decomon_symbolic_inputs = helpers.generate_merging_decomon_input_from_single_decomon_inputs(
        decomon_inputs=double_input(decomon_symbolic_inputs_0), ibp=ibp, affine=affine, propagation=propagation
    )

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
    # skip if empty affine bounds in forward propagation (it would generate issues to split inputs)
    if (
        decomon_layer.inputs_outputs_spec.cannot_have_empty_affine_inputs()
        and decomon_layer.inputs_outputs_spec.nb_input_tensors > len(decomon_symbolic_inputs)
    ):
        pytest.skip("Skip cases with empty (meaning identity) affine inputs that cannot be properly split")

    decomon_layer(decomon_symbolic_inputs)

    # call on actual inputs
    keras_model_input = keras_model_input_fn()
    keras_layer_input_0 = keras_layer_input_fn(keras_model_input)
    decomon_inputs_0 = decomon_input_fn(
        keras_model_input=keras_model_input, keras_layer_input=keras_layer_input_0, output_shape=output_shape
    )
    keras_layer_input = double_input(keras_layer_input_0)
    decomon_inputs = helpers.generate_merging_decomon_input_from_single_decomon_inputs(
        decomon_inputs=double_input(decomon_inputs_0), ibp=ibp, affine=affine, propagation=propagation
    )

    keras_output = layer(keras_layer_input)
    decomon_output = decomon_layer(decomon_inputs)

    # check affine representation is ok
    if decomon_layer.linear:
        w, b = decomon_layer.get_affine_representation()
        keras_output_2 = b
        for w_i, keras_layer_input_i in zip(w, keras_layer_input):
            diagonal = (False, w_i.shape == b.shape)
            missing_batchsize = (False, True)
            keras_output_2 = keras_output_2 + batch_multid_dot(
                keras_layer_input_i, w_i, missing_batchsize=missing_batchsize, diagonal=diagonal
            )  # += does not work well with broadcasting on pytorch backend
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
        is_merging_layer=True,
    )

    # before propagation through linear layer lower == upper => lower == upper after propagation
    if equal_bounds and is_actually_linear:
        helpers.assert_decomon_output_lower_equal_upper(
            decomon_output, ibp=ibp, affine=affine, propagation=propagation, decimal=decimal, is_merging_layer=True
        )
