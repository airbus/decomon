import keras.ops as K
import numpy as np
import pytest
from pytest_cases import parametrize

from decomon.core import BoxDomain, InputsOutputsSpec, Propagation
from decomon.layers.output import ConvertOutput


def generate_simple_inputs(
    ibp,
    affine,
    input_shape,
    output_shape,
    batchsize,
    diag,
    nobatch,
    perturbation_domain,
    needs_perturbation_domain_inputs=False,
):
    if ibp:
        x = K.convert_to_tensor(2.0 * np.random.random((batchsize,) + output_shape) - 1.0)
        lower = x - 0.5
        upper = x + 0.5
        constant_bounds = [lower, upper]
    else:
        constant_bounds = []
    if affine:
        if diag:
            w = K.ones(input_shape)
        else:
            w = K.reshape(K.eye(int(np.prod(input_shape)), int(np.prod(output_shape))), input_shape + output_shape)
        b = 0.1 * K.ones(output_shape)
        if not nobatch:
            w = K.repeat(w[None], batchsize, axis=0)
            b = K.repeat(b[None], batchsize, axis=0)
        affine_bounds = [w, -b, w, b]
    else:
        affine_bounds = []

    if needs_perturbation_domain_inputs:
        if isinstance(perturbation_domain, BoxDomain):
            keras_input = K.convert_to_tensor(2.0 * np.random.random((batchsize,) + input_shape) - 1.0)
            perturbation_domain_inputs = [
                K.concatenate([keras_input[:, None] - 0.1, keras_input[:, None] + 0.1], axis=1)
            ]
        else:
            raise NotImplementedError
    else:
        perturbation_domain_inputs = []

    return affine_bounds + constant_bounds + perturbation_domain_inputs


@parametrize("ibp_from, affine_from", [(True, False), (False, True), (True, True)], ids=["ibp", "affine", "hybrid"])
@parametrize("ibp_to, affine_to", [(True, False), (False, True), (True, True)], ids=["ibp", "affine", "hybrid"])
@parametrize(
    "diag, nobatch",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
    ids=["diagonal-nobatch", "diagonal", "nobatch", "generic"],
)
@parametrize("input_shape", [(3,), (5, 6, 2)], ids=["1d", "multid"])
@parametrize("output_shape", [(2,), (4, 5, 1)], ids=["1d", "multid"])
def test_convert_1output(
    ibp_from,
    affine_from,
    ibp_to,
    affine_to,
    diag,
    nobatch,
    input_shape,
    output_shape,
    batchsize,
    perturbation_domain,
    helpers,
):
    if not affine_from:
        if diag or nobatch:
            pytest.skip("diag and nobatch have no meaning when affine is False")
    if diag:
        if len(input_shape) != len(output_shape):
            pytest.skip("we need input_shape==output_shape if diag")
        else:
            output_shape = input_shape

    layer = ConvertOutput(
        perturbation_domain=perturbation_domain,
        ibp_from=ibp_from,
        affine_from=affine_from,
        ibp_to=ibp_to,
        affine_to=affine_to,
        model_output_shapes=[output_shape],
    )

    symbolic_input = helpers.get_decomon_symbolic_inputs(
        model_input_shape=input_shape,
        model_output_shape=None,
        layer_input_shape=output_shape,
        layer_output_shape=None,
        ibp=ibp_from,
        affine=affine_from,
        propagation=Propagation.FORWARD,
        perturbation_domain=perturbation_domain,
        diag=diag,
        nobatch=nobatch,
        remove_perturbation_domain_inputs=True,
        add_perturbation_domain_inputs=layer.needs_perturbation_domain_inputs(),
    )

    input = generate_simple_inputs(
        ibp=ibp_from,
        affine=affine_from,
        input_shape=input_shape,
        output_shape=output_shape,
        batchsize=batchsize,
        diag=diag,
        nobatch=nobatch,
        perturbation_domain=perturbation_domain,
        needs_perturbation_domain_inputs=layer.needs_perturbation_domain_inputs(),
    )

    symbolic_output = layer(symbolic_input)
    output = layer(input)

    # check shapes
    output_shape = [t.shape for t in output]
    expected_output_shape = [t.shape for t in symbolic_output]
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape

    inputs_outputs_spec_fused = InputsOutputsSpec(
        ibp=ibp_to, affine=layer.affine_to, layer_input_shape=layer.model_output_shapes, is_merging_layer=True
    )
    assert len(output_shape) == inputs_outputs_spec_fused.nb_output_tensors * layer.nb_outputs_keras_model


@parametrize("ibp_from, affine_from", [(True, False), (False, True), (True, True)], ids=["ibp", "affine", "hybrid"])
@parametrize("ibp_to, affine_to", [(True, False), (False, True), (True, True)], ids=["ibp", "affine", "hybrid"])
@parametrize(
    "diag_1, nobatch_1",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
    ids=["diagonal-nobatch", "diagonal", "nobatch", "generic"],
)
@parametrize("input_shape", [(3,), (5, 6, 2)], ids=["1d", "multid"])
@parametrize("output_shape_1", [(2,), (4, 5, 1)], ids=["1d", "multid"])
@parametrize("output_shape_2", [(3,), (2, 5, 1)], ids=["1d", "multid"])
def test_convert_2outputs(
    ibp_from,
    affine_from,
    ibp_to,
    affine_to,
    diag_1,
    nobatch_1,
    input_shape,
    output_shape_1,
    output_shape_2,
    batchsize,
    perturbation_domain,
    helpers,
):
    if not affine_from:
        if diag_1 or nobatch_1:
            pytest.skip("diag and nobatch have no meaning when affine is False")
    if diag_1:
        if len(input_shape) != len(output_shape_1):
            pytest.skip("we need input_shape==output_shape if diag")
        else:
            output_shape_1 = input_shape

    model_output_shapes = [output_shape_1, output_shape_2]
    layer = ConvertOutput(
        perturbation_domain=perturbation_domain,
        ibp_from=ibp_from,
        affine_from=affine_from,
        ibp_to=ibp_to,
        affine_to=affine_to,
        model_output_shapes=model_output_shapes,
    )

    symbolic_input_1 = helpers.get_decomon_symbolic_inputs(
        model_input_shape=input_shape,
        model_output_shape=None,
        layer_input_shape=output_shape_1,
        layer_output_shape=None,
        ibp=ibp_from,
        affine=affine_from,
        propagation=Propagation.FORWARD,
        perturbation_domain=perturbation_domain,
        diag=diag_1,
        nobatch=nobatch_1,
        remove_perturbation_domain_inputs=True,
    )
    symbolic_input_2 = helpers.get_decomon_symbolic_inputs(
        model_input_shape=input_shape,
        model_output_shape=None,
        layer_input_shape=output_shape_2,
        layer_output_shape=None,
        ibp=ibp_from,
        affine=affine_from,
        propagation=Propagation.FORWARD,
        perturbation_domain=perturbation_domain,
        remove_perturbation_domain_inputs=True,
        add_perturbation_domain_inputs=layer.needs_perturbation_domain_inputs(),
    )
    symbolic_input = symbolic_input_1 + symbolic_input_2

    input_1 = generate_simple_inputs(
        ibp=ibp_from,
        affine=affine_from,
        input_shape=input_shape,
        output_shape=output_shape_1,
        batchsize=batchsize,
        diag=diag_1,
        nobatch=nobatch_1,
        perturbation_domain=perturbation_domain,
    )
    input_2 = generate_simple_inputs(
        ibp=ibp_from,
        affine=affine_from,
        input_shape=input_shape,
        output_shape=output_shape_2,
        batchsize=batchsize,
        perturbation_domain=perturbation_domain,
        needs_perturbation_domain_inputs=layer.needs_perturbation_domain_inputs(),
        diag=False,
        nobatch=False,
    )
    input = input_1 + input_2

    symbolic_output = layer(symbolic_input)
    output = layer(input)

    # check shapes
    output_shape = [t.shape for t in output]
    expected_output_shape = [t.shape for t in symbolic_output]
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape

    inputs_outputs_spec_fused = InputsOutputsSpec(
        ibp=ibp_to, affine=layer.affine_to, layer_input_shape=layer.model_output_shapes, is_merging_layer=True
    )
    assert len(output_shape) == inputs_outputs_spec_fused.nb_output_tensors * layer.nb_outputs_keras_model
