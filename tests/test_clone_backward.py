# creating toy network and assess that the decomposition is correct
from __future__ import absolute_import

import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Add, Average, Dense, Input, Maximum
from tensorflow.keras.models import Model, Sequential

from decomon.backward_layers.backward_layers import get_backward
from decomon.layers.decomon_layers import DecomonAdd, DecomonDense, to_monotonic
from decomon.models.backward_cloning import get_backward_model as convert_backward
from decomon.models.forward_cloning import convert_forward

from . import (
    assert_output_properties_box,
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_standart_values_1d_box,
    get_tensor_decomposition_1d_box,
    get_tensor_decomposition_multid_box,
)
from .test_clone_forward import (
    dense_NN_1D,
    toy_network_tutorial,
    toy_struct_v0_1D,
    toy_struct_v1_1D,
    toy_struct_v2_1D,
)


@pytest.mark.parametrize(
    "n, archi, activation, sequential, use_bias, mode, use_input, floatx",
    [
        (0, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", True, 32),
        (3, [4, 3, 1], None, True, True, "forward", True, 32),
        (5, [4, 3, 1], None, True, True, "forward", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", True, 32),
        (3, [4, 3, 1], None, False, True, "forward", True, 32),
        (5, [4, 3, 1], None, False, True, "forward", True, 32),
        (0, [4, 3, 1], None, True, True, "ibp", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, True, True, "forward", True, 64),
        (1, [4, 3, 1], None, True, True, "forward", True, 64),
        (3, [4, 3, 1], None, True, True, "forward", True, 64),
        (5, [4, 3, 1], None, True, True, "forward", True, 64),
        (0, [4, 3, 1], None, False, True, "forward", True, 64),
        (1, [4, 3, 1], None, False, True, "forward", True, 64),
        (3, [4, 3, 1], None, False, True, "forward", True, 64),
        (5, [4, 3, 1], None, False, True, "forward", True, 64),
        (0, [4, 3, 1], None, True, True, "ibp", True, 64),
        (1, [4, 3, 1], None, True, True, "ibp", True, 64),
        (3, [4, 3, 1], None, True, True, "ibp", True, 64),
        (5, [4, 3, 1], None, True, True, "ibp", True, 64),
        (0, [4, 3, 1], None, False, True, "ibp", True, 64),
        (1, [4, 3, 1], None, False, True, "ibp", True, 64),
        (3, [4, 3, 1], None, False, True, "ibp", True, 64),
        (5, [4, 3, 1], None, False, True, "ibp", True, 64),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, True, True, "forward", True, 16),
        (1, [4, 3, 1], None, True, True, "forward", True, 16),
        (3, [4, 3, 1], None, True, True, "forward", True, 16),
        (5, [4, 3, 1], None, True, True, "forward", True, 16),
        (0, [4, 3, 1], None, False, True, "forward", True, 16),
        (1, [4, 3, 1], None, False, True, "forward", True, 16),
        (3, [4, 3, 1], None, False, True, "forward", True, 16),
        (5, [4, 3, 1], None, False, True, "forward", True, 16),
        (0, [4, 3, 1], None, True, True, "ibp", True, 16),
        (1, [4, 3, 1], None, True, True, "ibp", True, 16),
        (3, [4, 3, 1], None, True, True, "ibp", True, 16),
        (5, [4, 3, 1], None, True, True, "ibp", True, 16),
        (0, [4, 3, 1], None, False, True, "ibp", True, 16),
        (1, [4, 3, 1], None, False, True, "ibp", True, 16),
        (3, [4, 3, 1], None, False, True, "ibp", True, 16),
        (5, [4, 3, 1], None, False, True, "ibp", True, 16),
    ],
)
def test_convert_forward_1D(n, archi, activation, sequential, use_bias, mode, use_input, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_network_tutorial(dtype=K.floatx())
    ref_nn(inputs[1])

    IBP = True
    forward = True
    if mode == "forward":
        IBP = False
    if mode == "ibp":
        forward = False

    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]

    _, _, layer_map, forward_map = convert_forward(
        ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors, final_ibp=IBP, final_forward=forward
    )
    _, output, _, _ = convert_backward(
        ref_nn,
        input_tensors=input_tensors,
        IBP=IBP,
        forward=forward,
        layer_map=layer_map,
        forward_map=forward_map,
        final_ibp=IBP,
        final_forward=forward,
    )

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


"""



@pytest.mark.parametrize(
    "n, archi, activation, sequential, use_bias, mode, use_input, floatx",
    [
        (0, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", True, 32),
        (3, [4, 3, 1], None, True, True, "forward", True, 32),
        (5, [4, 3, 1], None, True, True, "forward", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", True, 32),
        (3, [4, 3, 1], None, False, True, "forward", True, 32),
        (5, [4, 3, 1], None, False, True, "forward", True, 32),
        (0, [4, 3, 1], None, True, True, "ibp", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, True, True, "forward", True, 64),
        (1, [4, 3, 1], None, True, True, "forward", True, 64),
        (3, [4, 3, 1], None, True, True, "forward", True, 64),
        (5, [4, 3, 1], None, True, True, "forward", True, 64),
        (0, [4, 3, 1], None, False, True, "forward", True, 64),
        (1, [4, 3, 1], None, False, True, "forward", True, 64),
        (3, [4, 3, 1], None, False, True, "forward", True, 64),
        (5, [4, 3, 1], None, False, True, "forward", True, 64),
        (0, [4, 3, 1], None, True, True, "ibp", True, 64),
        (1, [4, 3, 1], None, True, True, "ibp", True, 64),
        (3, [4, 3, 1], None, True, True, "ibp", True, 64),
        (5, [4, 3, 1], None, True, True, "ibp", True, 64),
        (0, [4, 3, 1], None, False, True, "ibp", True, 64),
        (1, [4, 3, 1], None, False, True, "ibp", True, 64),
        (3, [4, 3, 1], None, False, True, "ibp", True, 64),
        (5, [4, 3, 1], None, False, True, "ibp", True, 64),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, True, True, "forward", True, 16),
        (1, [4, 3, 1], None, True, True, "forward", True, 16),
        (3, [4, 3, 1], None, True, True, "forward", True, 16),
        (5, [4, 3, 1], None, True, True, "forward", True, 16),
        (0, [4, 3, 1], None, False, True, "forward", True, 16),
        (1, [4, 3, 1], None, False, True, "forward", True, 16),
        (3, [4, 3, 1], None, False, True, "forward", True, 16),
        (5, [4, 3, 1], None, False, True, "forward", True, 16),
        (0, [4, 3, 1], None, True, True, "ibp", True, 16),
        (1, [4, 3, 1], None, True, True, "ibp", True, 16),
        (3, [4, 3, 1], None, True, True, "ibp", True, 16),
        (5, [4, 3, 1], None, True, True, "ibp", True, 16),
        (0, [4, 3, 1], None, False, True, "ibp", True, 16),
        (1, [4, 3, 1], None, False, True, "ibp", True, 16),
        (3, [4, 3, 1], None, False, True, "ibp", True, 16),
        (5, [4, 3, 1], None, False, True, "ibp", True, 16),
    ],
)
def test_convert_backward_1D(n, archi, activation, sequential, use_bias, mode, use_input, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = dense_NN_1D(1, archi, sequential, activation, use_bias, dtype=K.floatx())
    ref_nn(inputs[1])

    IBP = True
    forward = True
    if mode == "forward":
        IBP = False
    if mode == "ibp":
        forward = False

    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]

    _, _, layer_map, forward_map = convert_forward(
        ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors
    )
    _, output, _, _ = convert_backward(
        ref_nn, input_tensors=input_tensors, IBP=IBP, forward=forward, layer_map=layer_map, forward_map=forward_map
    )

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)



@pytest.mark.parametrize(
    "n, archi, activation, sequential, use_bias, mode, use_input, floatx",
    [
        (0, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", True, 32),
        (3, [4, 3, 1], None, True, True, "forward", True, 32),
        (5, [4, 3, 1], None, True, True, "forward", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", True, 32),
        (3, [4, 3, 1], None, False, True, "forward", True, 32),
        (5, [4, 3, 1], None, False, True, "forward", True, 32),
        (0, [4, 3, 1], None, True, True, "ibp", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, True, True, "forward", True, 64),
        (1, [4, 3, 1], None, True, True, "forward", True, 64),
        (3, [4, 3, 1], None, True, True, "forward", True, 64),
        (5, [4, 3, 1], None, True, True, "forward", True, 64),
        (0, [4, 3, 1], None, False, True, "forward", True, 64),
        (1, [4, 3, 1], None, False, True, "forward", True, 64),
        (3, [4, 3, 1], None, False, True, "forward", True, 64),
        (5, [4, 3, 1], None, False, True, "forward", True, 64),
        (0, [4, 3, 1], None, True, True, "ibp", True, 64),
        (1, [4, 3, 1], None, True, True, "ibp", True, 64),
        (3, [4, 3, 1], None, True, True, "ibp", True, 64),
        (5, [4, 3, 1], None, True, True, "ibp", True, 64),
        (0, [4, 3, 1], None, False, True, "ibp", True, 64),
        (1, [4, 3, 1], None, False, True, "ibp", True, 64),
        (3, [4, 3, 1], None, False, True, "ibp", True, 64),
        (5, [4, 3, 1], None, False, True, "ibp", True, 64),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (3, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (5, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (3, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (5, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, True, True, "forward", True, 16),
        (1, [4, 3, 1], None, True, True, "forward", True, 16),
        (3, [4, 3, 1], None, True, True, "forward", True, 16),
        (5, [4, 3, 1], None, True, True, "forward", True, 16),
        (0, [4, 3, 1], None, False, True, "forward", True, 16),
        (1, [4, 3, 1], None, False, True, "forward", True, 16),
        (3, [4, 3, 1], None, False, True, "forward", True, 16),
        (5, [4, 3, 1], None, False, True, "forward", True, 16),
        (0, [4, 3, 1], None, True, True, "ibp", True, 16),
        (1, [4, 3, 1], None, True, True, "ibp", True, 16),
        (3, [4, 3, 1], None, True, True, "ibp", True, 16),
        (5, [4, 3, 1], None, True, True, "ibp", True, 16),
        (0, [4, 3, 1], None, False, True, "ibp", True, 16),
        (1, [4, 3, 1], None, False, True, "ibp", True, 16),
        (3, [4, 3, 1], None, False, True, "ibp", True, 16),
        (5, [4, 3, 1], None, False, True, "ibp", True, 16),
    ],
)
def test_convert_backward_rec_1D(n, archi, activation, sequential, use_bias, mode, use_input, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = dense_NN_1D(1, archi, sequential, activation, use_bias, dtype=K.floatx())
    ref_nn(inputs[1])

    IBP = True
    forward = True
    if mode == "forward":
        IBP = False
    if mode == "ibp":
        forward = False

    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]

    _, output, _, _ = convert_backward(ref_nn, input_tensors=input_tensors, x_tensor=z, IBP=IBP, forward=forward)

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


#### multi D


@pytest.mark.parametrize(
    "odd, archi, activation, sequential, use_bias, mode, use_input, floatx",
    [
        (0, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", True, 32),
        (0, [4, 3, 1], None, True, True, "ibp", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, True, True, "forward", True, 64),
        (1, [4, 3, 1], None, True, True, "forward", True, 64),
        (0, [4, 3, 1], None, False, True, "forward", True, 64),
        (1, [4, 3, 1], None, False, True, "forward", True, 64),
        (0, [4, 3, 1], None, True, True, "ibp", True, 64),
        (1, [4, 3, 1], None, True, True, "ibp", True, 64),
        (0, [4, 3, 1], None, False, True, "ibp", True, 64),
        (1, [4, 3, 1], None, False, True, "ibp", True, 64),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, True, True, "forward", True, 16),
        (1, [4, 3, 1], None, True, True, "forward", True, 16),
        (0, [4, 3, 1], None, False, True, "forward", True, 16),
        (1, [4, 3, 1], None, False, True, "forward", True, 16),
        (0, [4, 3, 1], None, True, True, "ibp", True, 16),
        (1, [4, 3, 1], None, True, True, "ibp", True, 16),
        (0, [4, 3, 1], None, False, True, "ibp", True, 16),
        (1, [4, 3, 1], None, False, True, "ibp", True, 16),
    ],
)
def test_convert_backward_multiD(odd, archi, activation, sequential, use_bias, mode, use_input, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = dense_NN_1D(y.shape[-1], archi, sequential, activation, use_bias, dtype=K.floatx())
    ref_nn(inputs[1])

    IBP = True
    forward = True
    if mode == "forward":
        IBP = False
    if mode == "ibp":
        forward = False

    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]

    f_ref = K.function(inputs, ref_nn(inputs[1]))
    y_ref = f_ref(inputs_)

    a_x, _, layer_map, forward_map = convert_forward(
        ref_nn, input_dim=x.shape[-1], IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors
    )
    b_x, output, _, _ = convert_backward(
        ref_nn,
        input_tensors=input_tensors,
        input_dim=x.shape[-1],
        x_tensor=z,
        IBP=IBP,
        forward=forward,
        layer_map=layer_map,
        forward_map=forward_map,
    )

    f_dense = K.function(inputs[2:], output)

    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6

    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)



@pytest.mark.parametrize(
    "odd, archi, activation, sequential, use_bias, mode, use_input, floatx",
    [
        (0, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", True, 32),
        (0, [4, 3, 1], None, True, True, "ibp", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 64),
        (0, [4, 3, 1], None, True, True, "forward", True, 64),
        (1, [4, 3, 1], None, True, True, "forward", True, 64),
        (0, [4, 3, 1], None, False, True, "forward", True, 64),
        (1, [4, 3, 1], None, False, True, "forward", True, 64),
        (0, [4, 3, 1], None, True, True, "ibp", True, 64),
        (1, [4, 3, 1], None, True, True, "ibp", True, 64),
        (0, [4, 3, 1], None, False, True, "ibp", True, 64),
        (1, [4, 3, 1], None, False, True, "ibp", True, 64),
        (0, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, True, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (1, [4, 3, 1], None, False, True, "hybrid", True, 16),
        (0, [4, 3, 1], None, True, True, "forward", True, 16),
        (1, [4, 3, 1], None, True, True, "forward", True, 16),
        (0, [4, 3, 1], None, False, True, "forward", True, 16),
        (1, [4, 3, 1], None, False, True, "forward", True, 16),
        (0, [4, 3, 1], None, True, True, "ibp", True, 16),
        (1, [4, 3, 1], None, True, True, "ibp", True, 16),
        (0, [4, 3, 1], None, False, True, "ibp", True, 16),
        (1, [4, 3, 1], None, False, True, "ibp", True, 16),
    ],
)
def test_convert_backward_rec_multiD(odd, archi, activation, sequential, use_bias, mode, use_input, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = dense_NN_1D(y.shape[-1], archi, sequential, activation, use_bias, dtype=K.floatx())
    ref_nn(inputs[1])

    IBP = True
    forward = True
    if mode == "forward":
        IBP = False
    if mode == "ibp":
        forward = False

    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]

    _, output, _, _ = convert_backward(ref_nn, input_tensors=input_tensors, x_tensor=z, IBP=IBP, forward=forward)

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
"""
#### SO FAR OK

"""
@pytest.mark.parametrize(
    "n, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (5, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (5, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (5, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (5, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (5, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (5, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (5, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (5, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (5, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (5, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (5, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
        (5, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_struct0_1D(n, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v0_1D(1, archi, activation, True)
    ref_nn(inputs[1])

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, _, layer_map, forward_map = convert_forward(
        ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors
    )

    _, output, _, _ = convert_backward(
        ref_nn,
        input_tensors=input_tensors,
        x_tensor=z,
        IBP=IBP,
        forward=forward,
        forward_map=forward_map,
    )


    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])
    import pdb; pdb.set_trace()

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)



@pytest.mark.parametrize(
    "n, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (5, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (5, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (5, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (5, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (5, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (5, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (5, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (5, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (5, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (5, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (5, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
        (5, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_rec_struct0_1D(n, archi, activation, use_input, mode, floatx, shared):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v0_1D(1, archi, activation, True)
    ref_nn(inputs[1])

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, output, _, _ = convert_backward(ref_nn, input_tensors=input_tensors, x_tensor=z, IBP=IBP, forward=forward)

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (5, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (5, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (5, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (5, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (5, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (5, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (5, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (5, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (5, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (5, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (5, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
        (5, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_struct1_1D(n, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v1_1D(1, archi, True, activation, True)
    ref_nn(inputs[1])

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, _, layer_map, forward_map = convert_forward(
        ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors
    )
    _, output, _, _ = convert_backward(
        ref_nn,
        input_tensors=input_tensors,
        x_tensor=z,
        IBP=IBP,
        forward=forward,
        layer_map=layer_map,
        forward_map=forward_map,
    )
    f_dense = K.function(inputs[2:], output)

    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (5, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (5, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (5, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (5, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (5, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (5, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (5, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (5, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (5, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (5, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (5, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
        (5, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_rec_struct1_1D(n, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v1_1D(1, archi, True, activation, True)
    ref_nn(inputs[1])

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, output, _, _ = convert_backward(ref_nn, input_tensors=input_tensors, IBP=IBP, forward=forward)
    f_dense = K.function(inputs[2:], output)

    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
"""

"""
@pytest.mark.parametrize(
    "n, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (5, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (5, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (5, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (5, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (5, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (5, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (5, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (5, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (5, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (5, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (5, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
        (5, [4, 1], "relu", False, "ibp", 32, True),
    ],
)

@pytest.mark.parametrize(
    "n, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
    ]
)
def test_convert_backward_struct2_1D(n, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v2_1D(1, archi, True, activation, True)
    ref_nn(inputs[1])

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, _, layer_map, forward_map = convert_forward(
        ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors
    )
    _, output, _, _ = convert_backward(
        ref_nn,
        input_tensors=input_tensors,
        x_tensor=z,
        IBP=IBP,
        forward=forward,
        layer_map=layer_map,
        forward_map=forward_map,
    )

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
"""


"""
@pytest.mark.parametrize(
    "n, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (5, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (5, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (5, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (5, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (5, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (5, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (5, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (5, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (5, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (5, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (5, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
        (5, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_rec_struct2_1D(n, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v2_1D(1, archi, True, activation, True)
    ref_nn(inputs[1])

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, output, _, _ = convert_backward(ref_nn, input_tensors=input_tensors, IBP=IBP, forward=forward)

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_struct1_multiD(odd, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v1_1D(y.shape[-1], archi, True, activation, True)
    ref_nn(inputs[1])
    f_ref = K.function(inputs, ref_nn(inputs[1]))
    y_ref = f_ref(inputs_)

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, _, layer_map, forward_map = convert_forward(
        ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors
    )
    _, output, _, _ = convert_backward(
        ref_nn,
        input_tensors=input_tensors,
        x_tensor=z,
        IBP=IBP,
        forward=forward,
        layer_map=layer_map,
        forward_map=forward_map,
    )
    f_dense = K.function(inputs[2:], output)

    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_rec_struct1_multiD(odd, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v1_1D(y.shape[-1], archi, True, activation, True)
    ref_nn(inputs[1])
    f_ref = K.function(inputs, ref_nn(inputs[1]))
    y_ref = f_ref(inputs_)

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, output, _, _ = convert_backward(ref_nn, input_tensors=input_tensors, IBP=IBP, forward=forward)
    f_dense = K.function(inputs[2:], output)

    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, archi, activation, use_input, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_rec_struct0_multiD(odd, archi, activation, use_input, mode, floatx, shared):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x_, y_, z_, u_c_, W_u_, b_u_, l_c_, W_l_, b_l_ = inputs_
    # inputs_ = [x_, y_, z_, u_c_, 0*W_u_, u_c_, l_c_, 0*W_l_, l_c_]

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v0_1D(y.shape[-1], archi, activation, True)
    ref_nn(inputs[1])

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, output, _, _ = convert_backward(ref_nn, input_tensors=input_tensors, IBP=IBP, forward=forward)

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
"""
"""
@pytest.mark.parametrize(
    "n, archi, activation, sequential, mode, floatx, shared",
    [
        (0, [4, 1], "relu", True, "hybrid", 32, False),
        (1, [4, 1], "relu", True, "hybrid", 32, False),
        (5, [4, 1], "relu", True, "hybrid", 32, False),
        (0, [4, 1], "relu", True, "forward", 32, False),
        (1, [4, 1], "relu", True, "forward", 32, False),
        (5, [4, 1], "relu", True, "forward", 32, False),
        (0, [4, 1], "relu", True, "ibp", 32, False),
        (1, [4, 1], "relu", True, "ibp", 32, False),
        (5, [4, 1], "relu", True, "ibp", 32, False),
        (0, [4, 1], "relu", False, "hybrid", 32, False),
        (1, [4, 1], "relu", False, "hybrid", 32, False),
        (5, [4, 1], "relu", False, "hybrid", 32, False),
        (0, [4, 1], "relu", False, "forward", 32, False),
        (1, [4, 1], "relu", False, "forward", 32, False),
        (5, [4, 1], "relu", False, "forward", 32, False),
        (0, [4, 1], "relu", False, "ibp", 32, False),
        (1, [4, 1], "relu", False, "ibp", 32, False),
        (5, [4, 1], "relu", False, "ibp", 32, False),
        (0, [4, 1], "relu", True, "hybrid", 32, True),
        (1, [4, 1], "relu", True, "hybrid", 32, True),
        (5, [4, 1], "relu", True, "hybrid", 32, True),
        (0, [4, 1], "relu", True, "forward", 32, True),
        (1, [4, 1], "relu", True, "forward", 32, True),
        (5, [4, 1], "relu", True, "forward", 32, True),
        (0, [4, 1], "relu", True, "ibp", 32, True),
        (1, [4, 1], "relu", True, "ibp", 32, True),
        (5, [4, 1], "relu", True, "ibp", 32, True),
        (0, [4, 1], "relu", False, "hybrid", 32, True),
        (1, [4, 1], "relu", False, "hybrid", 32, True),
        (5, [4, 1], "relu", False, "hybrid", 32, True),
        (0, [4, 1], "relu", False, "forward", 32, True),
        (1, [4, 1], "relu", False, "forward", 32, True),
        (5, [4, 1], "relu", False, "forward", 32, True),
        (0, [4, 1], "relu", False, "ibp", 32, True),
        (1, [4, 1], "relu", False, "ibp", 32, True),
        (5, [4, 1], "relu", False, "ibp", 32, True),
    ],
)
def test_convert_backward_stack_1D(n, archi, activation, sequential, mode, floatx, shared):

    use_input = True
    use_bias = True
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn_0 = dense_NN_1D(1, archi[:-1], sequential, activation, use_bias, dtype=K.floatx())
    ref_nn_1 = dense_NN_1D(archi[-2], archi[-1:], sequential, activation, use_bias, dtype=K.floatx())
    if sequential:
        ref_nn = Sequential([ref_nn_0, ref_nn_1])
    else:
        ref_nn = Model(inputs[1], ref_nn_1(ref_nn_0(inputs[1])))
    ref_nn(inputs[1])

    IBP = True
    forward = True
    input_tensors = inputs[2:]
    if mode == "ibp":
        input_tensors = [u_c, l_c]
        forward = False
    if mode == "forward":
        input_tensors = [z, W_u, b_u, W_l, b_l]
        forward = True
        IBP = False

    _, _, _, forward_map = convert_forward(
        ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors
    )
    _, output, _, _ = convert_backward(
        ref_nn,
        input_tensors=input_tensors,
        x_tensor=z,
        IBP=IBP,
        forward=forward,
        forward_map=forward_map,
    )

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode == "forward":

        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
"""
