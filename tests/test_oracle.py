from typing import TypeVar

import pytest
from pytest_cases import parametrize

from decomon.constants import Propagation
from decomon.layers.oracle import DecomonOracle

T = TypeVar("T")


def double_input(input: T) -> list[T]:
    return [input] * 2


@parametrize("is_merging_layer", [False, True])
def test_decomon_oracle(
    ibp,
    affine,
    propagation,
    perturbation_domain,
    is_merging_layer,
    input_shape,
    empty,
    batchsize,
    simple_decomon_symbolic_input_fn,
    simple_decomon_input_fn,
    simple_keras_model_input_fn,
    simple_keras_layer_input_fn,
    helpers,
):
    if propagation == Propagation.BACKWARD:
        pytest.skip(
            "We skip backward propagation as we need same inputs for DecomonOracle as for a forward layer, even for crown."
        )
    if empty:
        pytest.skip("Empty inputs without meaning for DecomonOracle.")

    output_shape = None
    linear = False

    decomon_symbolic_input = simple_decomon_symbolic_input_fn(output_shape, linear)
    keras_model_input = simple_keras_model_input_fn()
    keras_layer_input = simple_keras_layer_input_fn(keras_model_input)
    decomon_input = simple_decomon_input_fn(
        keras_model_input=keras_model_input,
        keras_layer_input=keras_layer_input,
        output_shape=output_shape,
        linear=linear,
    )

    if is_merging_layer:
        decomon_symbolic_input = helpers.generate_merging_decomon_input_from_single_decomon_inputs(
            decomon_inputs=double_input(decomon_symbolic_input),
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            linear=linear,
        )
        decomon_input = helpers.generate_merging_decomon_input_from_single_decomon_inputs(
            decomon_inputs=double_input(decomon_input),
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            linear=linear,
        )
        input_shape = double_input(input_shape)

    layer = DecomonOracle(
        perturbation_domain=perturbation_domain,
        ibp=ibp,
        affine=affine,
        is_merging_layer=is_merging_layer,
        layer_input_shape=input_shape,
    )
    symbolic_output = layer(decomon_symbolic_input)
    output = layer(decomon_input)

    # check shapes
    if is_merging_layer:
        output_shape = [[t.shape for t in output_i] for output_i in output]
        expected_output_shape = [
            helpers.replace_none_by_batchsize(shapes=[t.shape for t in symbolic_output_i], batchsize=batchsize)
            for symbolic_output_i in symbolic_output
        ]
        assert output_shape == expected_output_shape

    else:
        output_shape = [t.shape for t in output]
        expected_output_shape = [t.shape for t in symbolic_output]
        expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
        assert output_shape == expected_output_shape
