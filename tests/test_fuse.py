import keras.ops as K
import numpy as np
import pytest
from pytest_cases import parametrize

from decomon.constants import Propagation
from decomon.layers.fuse import Fuse
from decomon.layers.inputs_outputs_specs import InputsOutputsSpec


def generate_simple_inputs(ibp, affine, input_shape, output_shape, batchsize, diag, nobatch):
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

    return affine_bounds + constant_bounds


@parametrize("ibp1, affine1", [(True, False), (False, True), (True, True)], ids=["ibp", "affine", "hybrid"])
@parametrize("ibp2, affine2", [(True, False), (False, True), (True, True)], ids=["ibp", "affine", "hybrid"])
@parametrize(
    "diag1, nobatch1",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
    ids=["diagonal-nobatch", "diagonal", "nobatch", "generic"],
)
@parametrize(
    "diag2, nobatch2",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
    ids=["diagonal-nobatch", "diagonal", "nobatch", "generic"],
)
@parametrize("input_shape1", [(3,), (5, 6, 2)], ids=["1d", "multid"])
@parametrize("output_shape1", [(2,), (4, 5, 1)], ids=["1d", "multid"])
@parametrize("output_shape2", [(4,), (3, 4, 3)], ids=["1d", "multid"])
def test_fuse_1output(
    ibp1,
    affine1,
    ibp2,
    affine2,
    diag1,
    nobatch1,
    diag2,
    nobatch2,
    input_shape1,
    output_shape1,
    output_shape2,
    batchsize,
    helpers,
):
    if not affine1:
        if diag1 or nobatch1:
            pytest.skip("diag and nobatch have no meaning when affine is False")
    if not affine2:
        if diag2 or nobatch2:
            pytest.skip("diag and nobatch have no meaning when affine is False")
    if diag1:
        if len(input_shape1) != len(output_shape1):
            pytest.skip("we need input_shape==output_shape if diag")
        else:
            output_shape1 = input_shape1
    if diag2:
        if len(output_shape1) != len(output_shape2):
            pytest.skip("we need input_shape==output_shape if diag")
        else:
            output_shape2 = output_shape1

    input_shape2 = output_shape1

    symbolic_input_1 = helpers.get_decomon_symbolic_inputs(
        model_input_shape=input_shape1,
        model_output_shape=None,
        layer_input_shape=output_shape1,
        layer_output_shape=None,
        ibp=ibp1,
        affine=affine1,
        propagation=Propagation.FORWARD,
        perturbation_domain=None,
        diag=diag1,
        nobatch=nobatch1,
        remove_perturbation_domain_inputs=True,
    )
    symbolic_input_2 = helpers.get_decomon_symbolic_inputs(
        model_input_shape=input_shape2,
        model_output_shape=None,
        layer_input_shape=output_shape2,
        layer_output_shape=None,
        ibp=ibp2,
        affine=affine2,
        propagation=Propagation.FORWARD,
        perturbation_domain=None,
        diag=diag2,
        nobatch=nobatch2,
        remove_perturbation_domain_inputs=True,
    )

    input_1 = generate_simple_inputs(
        ibp=ibp1,
        affine=affine1,
        input_shape=input_shape1,
        output_shape=output_shape1,
        batchsize=batchsize,
        diag=diag1,
        nobatch=nobatch1,
    )
    input_2 = generate_simple_inputs(
        ibp=ibp2,
        affine=affine2,
        input_shape=input_shape2,
        output_shape=output_shape2,
        batchsize=batchsize,
        diag=diag2,
        nobatch=nobatch2,
    )

    layer = Fuse(
        ibp_1=ibp1,
        affine_1=affine1,
        ibp_2=ibp2,
        affine_2=affine2,
        m1_input_shape=input_shape1,
        m_1_output_shapes=[output_shape1],
        from_linear_2=[nobatch2],
    )

    symbolic_output = layer((symbolic_input_1, symbolic_input_2))
    output = layer((input_1, input_2))

    # check shapes
    output_shape = [t.shape for t in output]
    expected_output_shape = [t.shape for t in symbolic_output]
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape

    inputs_outputs_spec_fused = InputsOutputsSpec(
        ibp=layer.ibp_fused, affine=layer.affine_fused, layer_input_shape=input_shape1
    )
    assert len(output_shape) == inputs_outputs_spec_fused.nb_output_tensors


@parametrize(
    "diag1_1, nobatch1_1",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
    ids=["diagonal-nobatch", "diagonal", "nobatch", "generic"],
)
@parametrize(
    "diag2_2, nobatch2_2",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
    ids=["diagonal-nobatch", "diagonal", "nobatch", "generic"],
)
@parametrize("input_shape1", [(3,), (5, 6, 2)], ids=["1d", "multid"])
@parametrize("output_shape1_1", [(2,), (4, 5, 1)], ids=["1d", "multid"])
@parametrize("output_shape1_2", [(3,), (2, 5, 1)], ids=["1d", "multid"])
@parametrize("output_shape2", [(4,), (3, 4, 3)], ids=["1d", "multid"])
def test_fuse_2outputs(
    diag1_1,
    nobatch1_1,
    diag2_2,
    nobatch2_2,
    input_shape1,
    output_shape1_1,
    output_shape1_2,
    output_shape2,
    batchsize,
    helpers,
):
    if diag1_1:
        if len(input_shape1) != len(output_shape1_1):
            pytest.skip("we need input_shape==output_shape if diag")
        else:
            output_shape1_1 = input_shape1
    if diag2_2:
        if len(output_shape1_2) != len(output_shape2):
            pytest.skip("we need input_shape==output_shape if diag")
        else:
            output_shape2 = output_shape1_2

    ibp1, affine1, ibp2, affine2 = True, True, False, True
    diag1_2, nobatch1_2 = False, False
    diag2_1, nobatch2_1 = False, False
    diag1 = [diag1_1, diag1_2]
    diag2 = [diag2_1, diag2_2]
    nobatch1 = [nobatch1_1, nobatch1_2]
    nobatch2 = [nobatch2_1, nobatch2_2]
    output_shape1 = [output_shape1_1, output_shape1_2]
    input_shape2 = output_shape1
    nb_m1_outputs = len(output_shape1)

    symbolic_input_1 = []
    for output_shape1_i, diag1_i, nobatch1_i in zip(output_shape1, diag1, nobatch1):
        symbolic_input_1 += helpers.get_decomon_symbolic_inputs(
            model_input_shape=input_shape1,
            model_output_shape=None,
            layer_input_shape=output_shape1_i,
            layer_output_shape=None,
            ibp=ibp1,
            affine=affine1,
            propagation=Propagation.FORWARD,
            perturbation_domain=None,
            diag=diag1_i,
            nobatch=nobatch1_i,
            remove_perturbation_domain_inputs=True,
        )
    symbolic_input_2 = []
    for input_shape2_i, diag2_i, nobatch2_i in zip(input_shape2, diag2, nobatch2):
        symbolic_input_2 += helpers.get_decomon_symbolic_inputs(
            model_input_shape=input_shape2_i,
            model_output_shape=None,
            layer_input_shape=output_shape2,
            layer_output_shape=None,
            ibp=ibp2,
            affine=affine2,
            propagation=Propagation.FORWARD,
            perturbation_domain=None,
            diag=diag2_i,
            nobatch=nobatch2_i,
            remove_perturbation_domain_inputs=True,
        )

    input_1 = []
    for output_shape1_i, diag1_i, nobatch1_i in zip(output_shape1, diag1, nobatch1):
        input_1 += generate_simple_inputs(
            ibp=ibp1,
            affine=affine1,
            input_shape=input_shape1,
            output_shape=output_shape1_i,
            batchsize=batchsize,
            diag=diag1_i,
            nobatch=nobatch1_i,
        )
    input_2 = []
    for input_shape2_i, diag2_i, nobatch2_i in zip(input_shape2, diag2, nobatch2):
        input_2 += generate_simple_inputs(
            ibp=ibp2,
            affine=affine2,
            input_shape=input_shape2_i,
            output_shape=output_shape2,
            batchsize=batchsize,
            diag=diag2_i,
            nobatch=nobatch2_i,
        )

    layer = Fuse(
        ibp_1=ibp1,
        affine_1=affine1,
        ibp_2=ibp2,
        affine_2=affine2,
        m1_input_shape=input_shape1,
        m_1_output_shapes=output_shape1,
        from_linear_2=nobatch2,
    )

    symbolic_output = layer((symbolic_input_1, symbolic_input_2))
    output = layer((input_1, input_2))

    # check shapes
    output_shape = [t.shape for t in output]
    expected_output_shape = [t.shape for t in symbolic_output]
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape

    inputs_outputs_spec_fused = InputsOutputsSpec(
        ibp=layer.ibp_fused, affine=layer.affine_fused, layer_input_shape=input_shape1
    )
    assert len(output_shape) == inputs_outputs_spec_fused.nb_output_tensors * nb_m1_outputs
