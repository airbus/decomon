import pytest
from pytest_cases import parametrize

from decomon.core import Propagation
from decomon.layers.input import ForwardInput


def test_forward_input(
    ibp,
    affine,
    propagation,
    perturbation_domain,
    simple_model_decomon_symbolic_input,
    simple_model_decomon_input,
    batchsize,
    helpers,
):
    if propagation == Propagation.BACKWARD:
        pytest.skip("backward propagation meaningless for ForwardInput")

    layer = ForwardInput(perturbation_domain=perturbation_domain, ibp=ibp, affine=affine)
    symbolic_output = layer(simple_model_decomon_symbolic_input)
    output = layer(simple_model_decomon_input)

    # check shapes
    output_shape = [t.shape for t in output]
    expected_output_shape = [t.shape for t in symbolic_output]
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape
