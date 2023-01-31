# creating toy network and assess that the decomposition is correct


import pytest
import tensorflow.python.keras.backend as K

from decomon.models.backward_cloning import get_backward_model as convert_backward
from decomon.models.forward_cloning import convert_forward


def test_convert_backward_1D(n, mode, floatx, helpers):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
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

    helpers.assert_output_properties_box(
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
