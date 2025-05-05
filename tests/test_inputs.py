import pytest
from pytest_cases import parametrize

from decomon.constants import Propagation
from decomon.layers.input import BackwardInput, ForwardInput


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


@parametrize("m1_output_shape", [(2,), (4, 5, 1)], ids=["1d", "multid"])
@parametrize("m2_output_shape", [(4,), (3, 4, 3)], ids=["1d", "multid"])
@parametrize("bb_nb_tensors", [1, 2, 3, 4, 8])
@parametrize("nb_outputs", [1, 2])
def test_backward_input(
    empty, diag, nobatch, bb_nb_tensors, m1_output_shape, m2_output_shape, nb_outputs, batchsize, helpers
):
    if empty:
        pytest.skip("backward bounds cannot be empty for BackwardInput")
    if diag:
        if len(m1_output_shape) != len(m2_output_shape):
            pytest.skip("we need output_shape1==output_shape2 if diag")
        else:
            m2_output_shape = m1_output_shape
    if bb_nb_tensors > 4 and nb_outputs == 1:
        pytest.skip("no meaning for more than 4 tensors if only 1 output to keras model")

    # for other outputs after the first
    if bb_nb_tensors <= 4:
        # same output shapes
        m1_output_shape_2 = m1_output_shape
        m2_output_shape_2 = m2_output_shape
    else:
        # can be different
        m1_output_shape_2 = (2, 3)
        m2_output_shape_2 = (3,)

    from_linear = [nobatch] + (nb_outputs - 1) * [False]
    model_output_shapes = [m1_output_shape] + (nb_outputs - 1) * [m1_output_shape_2]
    layer = BackwardInput(model_output_shapes=model_output_shapes, from_linear=from_linear)

    symbolic_input = helpers.get_decomon_symbolic_inputs(
        model_input_shape=m1_output_shape,
        model_output_shape=m2_output_shape,
        layer_input_shape=m2_output_shape,
        layer_output_shape=m2_output_shape,
        perturbation_domain=None,
        ibp=False,
        affine=True,
        propagation=Propagation.FORWARD,
        empty=empty,
        diag=diag,
        nobatch=nobatch,
        remove_perturbation_domain_inputs=True,
    )
    for model_output_shape in model_output_shapes[1:]:
        symbolic_input += helpers.get_decomon_symbolic_inputs(
            model_input_shape=model_output_shape,
            model_output_shape=m2_output_shape_2,
            layer_input_shape=m2_output_shape_2,
            layer_output_shape=m2_output_shape_2,
            perturbation_domain=None,
            ibp=False,
            affine=True,
            propagation=Propagation.FORWARD,
            remove_perturbation_domain_inputs=True,
        )
    symbolic_input = symbolic_input[:bb_nb_tensors]
    if nobatch:
        # nobatch only for 4 first tensors
        random_input = [
            helpers.generate_random_tensor(t.shape, batchsize=batchsize, nobatch=nobatch) for t in symbolic_input[:4]
        ]
        # add batchsize for following tensors
        random_input += [
            helpers.generate_random_tensor(t.shape[1:], batchsize=batchsize, nobatch=False) for t in symbolic_input[4:]
        ]
    else:
        random_input = [
            helpers.generate_random_tensor(t.shape[1:], batchsize=batchsize, nobatch=nobatch) for t in symbolic_input
        ]

    if bb_nb_tensors == 3:
        with pytest.raises(ValueError):
            symbolic_output = layer(symbolic_input)
    else:
        symbolic_output = layer(symbolic_input)
        output = layer(random_input)

        # check shapes
        output_shape = [t.shape for t in output]
        expected_output_shape = [t.shape for t in symbolic_output]
        expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
        assert output_shape == expected_output_shape
