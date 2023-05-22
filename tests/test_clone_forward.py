# creating toy network and assess that the decomposition is correct


import numpy as np
import pytest
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    Average,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model, Sequential

from decomon.layers.core import ForwardMode
from decomon.models.forward_cloning import convert_forward


# @pytest.mark.parametrize(
#    "n, activation, mode, shared, floatx",
#
def test_toy_network_1D(helpers, n, archi=None, activation="relu", use_bias=True):

    if archi is None:
        archi = [4, 1]
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standard_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    seq_nn = helpers.dense_NN_1D(1, archi, True, activation, use_bias)
    model_nn = helpers.dense_NN_1D(1, archi, False, activation, use_bias)

    _ = seq_nn.predict(y)
    _ = model_nn.predict(y)
    model_nn.set_weights(seq_nn.get_weights())

    y_0 = seq_nn.predict(y)
    y_1 = model_nn.predict(y)

    np.allclose(y_0, y_1)


def test_toy_network_multiD(helpers, odd, archi=None, activation="relu", use_bias=True):

    if archi is None:
        archi = [4, 1]
    inputs = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_ = helpers.get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    seq_nn = helpers.dense_NN_1D(y.shape[-1], archi, True, activation, use_bias)
    model_nn = helpers.dense_NN_1D(y.shape[-1], archi, False, activation, use_bias)

    _ = seq_nn.predict(y)
    _ = model_nn.predict(y)
    model_nn.set_weights(seq_nn.get_weights())

    y_0 = seq_nn.predict(y)
    y_1 = model_nn.predict(y)

    np.allclose(y_0, y_1)


# network from tutorial 1
def test_convert_forward_1D(n, mode, floatx, helpers):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_, y_, z_, u_c_, W_u_, b_u_, l_c_, W_l_, b_l_ = inputs_

    # We force W & b to be identity and zeros
    b_u_ = np.zeros_like(b_u_)
    b_l_ = np.zeros_like(b_l_)
    W_u_ = np.repeat(np.identity(n=W_u_.shape[-1])[None, :, :], repeats=W_u_.shape[0], axis=0)
    W_l_ = np.repeat(np.identity(n=W_l_.shape[-1])[None, :, :], repeats=W_l_.shape[0], axis=0)
    inputs_ = x_, y_, z_, u_c_, W_u_, b_u_, l_c_, W_l_, b_l_

    input_ref = y_
    input_box_tensor = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)

    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    output_ref = ref_nn.predict(input_ref)

    ibp = True
    affine = True
    mode = ForwardMode(mode)
    if mode == ForwardMode.AFFINE:
        ibp = False
    if mode == ForwardMode.IBP:
        affine = False

    input_tensors = [input_box_tensor, u_c, W_u, b_u, l_c, W_l, b_l]
    if mode == ForwardMode.IBP:
        input_tensors = [u_c, l_c]
    if mode == ForwardMode.AFFINE:
        input_tensors = [input_box_tensor, W_u, b_u, W_l, b_l]

    _, output, _, _ = convert_forward(ref_nn, ibp=ibp, affine=affine, shared=True, input_tensors=input_tensors)

    f_decomon = K.function(inputs, output)

    u_c_model, w_u_model, b_u_model, l_c_model, w_l_model, b_l_model = [None] * 6
    if mode == ForwardMode.HYBRID:
        z_model, u_c_model, w_u_model, b_u_model, l_c_model, w_l_model, b_l_model = f_decomon(inputs_)
    elif mode == ForwardMode.IBP:
        u_c_model, l_c_model = f_decomon(inputs_)
    elif mode == ForwardMode.AFFINE:
        z_model, w_u_model, b_u_model, w_l_model, b_l_model = f_decomon(inputs_)
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
        input_ref,
        output_ref,
        None,
        None,
        l_c_,
        u_c_,
        u_c_model,
        w_u_model,
        b_u_model,
        l_c_model,
        w_l_model,
        b_l_model,
        "convert_forward_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
