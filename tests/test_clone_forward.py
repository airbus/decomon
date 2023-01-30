# creating toy network and assess that the decomposition is correct


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
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

from decomon.models.forward_cloning import convert_forward


def dense_NN_1D(input_dim, archi, sequential, activation, use_bias, dtype="float32"):

    layers = [Dense(archi[0], use_bias=use_bias, activation=activation, input_dim=input_dim, dtype=dtype)]
    layers += [Dense(n_i, use_bias=use_bias, activation=activation, dtype=dtype) for n_i in archi[1:]]

    if sequential:
        return Sequential(layers)
    else:
        x = Input(input_dim, dtype=dtype)
        output = layers[0](x)
        for layer_ in layers[1:]:
            output = layer_(output)
        return Model(x, output)


def toy_struct_v0_1D(input_dim, archi, activation, use_bias, merge_op="average", dtype="float32"):

    nnet_0 = dense_NN_1D(input_dim, archi, False, activation, use_bias, dtype=dtype)
    # nnet_1 = dense_NN_1D(input_dim, archi, False, activation, use_bias)
    # nnet_0 = Dense(archi[-1], use_bias=use_bias, activation=activation, input_dim=input_dim)
    # nnet_1 = dense_NN_1D(input_dim, archi[-1:], True, activation, use_bias)
    nnet_1 = Dense(archi[-1], use_bias=use_bias, activation="linear", input_dim=input_dim, name="toto", dtype=dtype)
    nnet_2 = Dense(archi[-1], use_bias=use_bias, activation="linear", input_dim=input_dim, name="titi", dtype=dtype)

    # nnet_0 = Activation('linear')

    x = Input(input_dim, dtype=dtype)
    h_0 = nnet_0(x)
    h_1 = nnet_1(x)

    merge_op = "add"
    if merge_op == "average":
        y = Average(dtype=dtype)([h_0, h_1])
    if merge_op == "add":
        y = Add(dtype=dtype)([h_0, h_1])

    return Model(x, y)


def toy_struct_v1_1D(input_dim, archi, sequential, activation, use_bias, merge_op="average", dtype="float32"):

    nnet_0 = dense_NN_1D(input_dim, archi, sequential, activation, use_bias, dtype=dtype)

    x = Input(input_dim, dtype=dtype)
    h_0 = nnet_0(x)
    h_1 = nnet_0(x)
    if merge_op == "average":
        y = Average(dtype=dtype)([h_0, h_1])
    if merge_op == "add":
        y = Add(dtype=dtype)([h_0, h_1])

    return Model(x, y)


def toy_struct_v2_1D(input_dim, archi, sequential, activation, use_bias, merge_op="average", dtype="float32"):

    nnet_0 = dense_NN_1D(input_dim, archi, sequential, activation, use_bias, dtype=dtype)
    nnet_1 = dense_NN_1D(input_dim, archi, sequential, activation, use_bias, dtype=dtype)
    nnet_2 = Dense(archi[-1], use_bias=use_bias, activation="linear", input_dim=input_dim, dtype=dtype)

    x = Input(input_dim, dtype=dtype)
    nnet_0(x)
    nnet_1(x)
    nnet_1.set_weights([-p for p in nnet_0.get_weights()])  # be sure that the produced output will differ
    h_0 = nnet_2(nnet_0(x))
    h_1 = nnet_2(nnet_1(x))
    if merge_op == "average":
        y = Average(dtype=dtype)([h_0, h_1])
    if merge_op == "add":
        y = Add(dtype=dtype)([h_0, h_1])

    return Model(x, y)


def toy_struct_cnn(dtype="float32"):

    layers = [
        Conv2D(10, kernel_size=(3, 3), activation="relu", data_format="channels_last", dtype=dtype),
        Flatten(dtype=dtype),
        Dense(1, dtype=dtype),
    ]
    return Sequential(layers)


# @pytest.mark.parametrize(
#    "n, activation, mode, shared, floatx",
#
def test_toy_network_1D(helpers, n=0, archi=None, activation="relu", use_bias=True):

    if archi is None:
        archi = [4, 1]
    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standart_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    seq_nn = dense_NN_1D(1, archi, True, activation, use_bias)
    model_nn = dense_NN_1D(1, archi, False, activation, use_bias)

    _ = seq_nn.predict(y)
    _ = model_nn.predict(y)
    model_nn.set_weights(seq_nn.get_weights())

    y_0 = seq_nn.predict(y)
    y_1 = model_nn.predict(y)

    np.allclose(y_0, y_1)


def test_toy_network_multiD(helpers, odd=0, archi=None, activation="relu", use_bias=True):

    if archi is None:
        archi = [4, 1]
    inputs = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_ = helpers.get_standard_values_multid_box(odd)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    seq_nn = dense_NN_1D(y.shape[-1], archi, True, activation, use_bias)
    model_nn = dense_NN_1D(y.shape[-1], archi, False, activation, use_bias)

    _ = seq_nn.predict(y)
    _ = model_nn.predict(y)
    model_nn.set_weights(seq_nn.get_weights())

    y_0 = seq_nn.predict(y)
    y_1 = model_nn.predict(y)

    np.allclose(y_0, y_1)


# network from tutorial 1
def test_convert_forward_1D(n, mode, floatx, helpers):
    archi = [4, 3, 1]

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standart_values_1d_box(n, dc_decomp=False)
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

    _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors)

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
