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


"""
@pytest.mark.parametrize("n, archi, activation, sequential, use_bias, mode, use_input, floatx", [
    (0, [4, 3, 1], None, True, True, "hybrid", True,32), (1, [4, 3, 1], None, True, True, "hybrid", True,32), (3, [4, 3, 1], None, True, True, "hybrid", True,32), (5, [4, 3, 1], None, True, True, "hybrid", True,32),
    (0, [4, 3, 1], None, False, True, "hybrid", True,32), (1, [4, 3, 1], None, False, True, "hybrid", True,32), (3, [4, 3, 1], None, False, True, "hybrid", True,32), (5, [4, 3, 1], None, False, True, "hybrid", True,32),
    (0, [4, 3, 1], None, True, True, "forward", True,32), (1, [4, 3, 1], None, True, True, "forward", True,32), (3, [4, 3, 1], None, True, True, "forward", True,32), (5, [4, 3, 1], None, True, True, "forward", True,32),
    (0, [4, 3, 1], None, False, True, "forward", True,32), (1, [4, 3, 1], None, False, True, "forward", True,32), (3, [4, 3, 1], None, False, True, "forward", True,32), (5, [4, 3, 1], None, False, True, "forward", True,32),
    (0, [4, 3, 1], None, True, True, "ibp", True,32), (1, [4, 3, 1], None, True, True, "ibp", True,32), (3, [4, 3, 1], None, True, True, "ibp", True,32), (5, [4, 3, 1], None, True, True, "ibp", True,32),
    (0, [4, 3, 1], None, False, True, "ibp", True,32), (1, [4, 3, 1], None, False, True, "ibp", True,32), (3, [4, 3, 1], None, False, True, "ibp", True,32), (5, [4, 3, 1], None, False, True, "ibp", True,32),
    (0, [4, 3, 1], None, True, True, "hybrid", True,64), (1, [4, 3, 1], None, True, True, "hybrid", True,64), (3, [4, 3, 1], None, True, True, "hybrid", True,64), (5, [4, 3, 1], None, True, True, "hybrid", True,64),
    (0, [4, 3, 1], None, False, True, "hybrid", True,64), (1, [4, 3, 1], None, False, True, "hybrid", True,64), (3, [4, 3, 1], None, False, True, "hybrid", True,64), (5, [4, 3, 1], None, False, True, "hybrid", True,64),
    (0, [4, 3, 1], None, True, True, "forward", True,64), (1, [4, 3, 1], None, True, True, "forward", True,64), (3, [4, 3, 1], None, True, True, "forward", True,64), (5, [4, 3, 1], None, True, True, "forward", True,64),
    (0, [4, 3, 1], None, False, True, "forward", True,64), (1, [4, 3, 1], None, False, True, "forward", True,64), (3, [4, 3, 1], None, False, True, "forward", True,64), (5, [4, 3, 1], None, False, True, "forward", True,64),
    (0, [4, 3, 1], None, True, True, "ibp", True,64), (1, [4, 3, 1], None, True, True, "ibp", True,64), (3, [4, 3, 1], None, True, True, "ibp", True,64), (5, [4, 3, 1], None, True, True, "ibp", True,64),
    (0, [4, 3, 1], None, False, True, "ibp", True,64), (1, [4, 3, 1], None, False, True, "ibp", True,64), (3, [4, 3, 1], None, False, True, "ibp", True,64), (5, [4, 3, 1], None, False, True, "ibp", True,64),
    (0, [4, 3, 1], None, True, True, "hybrid", True,16), (1, [4, 3, 1], None, True, True, "hybrid", True,16), (3, [4, 3, 1], None, True, True, "hybrid", True,16), (5, [4, 3, 1], None, True, True, "hybrid", True,16),
    (0, [4, 3, 1], None, False, True, "hybrid", True,16), (1, [4, 3, 1], None, False, True, "hybrid", True,16), (3, [4, 3, 1], None, False, True, "hybrid", True,16), (5, [4, 3, 1], None, False, True, "hybrid", True,16),
    (0, [4, 3, 1], None, True, True, "forward", True,16), (1, [4, 3, 1], None, True, True, "forward", True,16), (3, [4, 3, 1], None, True, True, "forward", True,16), (5, [4, 3, 1], None, True, True, "forward", True,16),
    (0, [4, 3, 1], None, False, True, "forward", True,16), (1, [4, 3, 1], None, False, True, "forward", True,16), (3, [4, 3, 1], None, False, True, "forward", True,16), (5, [4, 3, 1], None, False, True, "forward", True,16),
    (0, [4, 3, 1], None, True, True, "ibp", True,16), (1, [4, 3, 1], None, True, True, "ibp", True,16), (3, [4, 3, 1], None, True, True, "ibp", True,16), (5, [4, 3, 1], None, True, True, "ibp", True,16),
    (0, [4, 3, 1], None, False, True, "ibp", True,16), (1, [4, 3, 1], None, False, True, "ibp", True,16), (3, [4, 3, 1], None, False, True, "ibp", True,16), (5, [4, 3, 1], None, False, True, "ibp", True,16),
])
def test_convert_forward_1D(n, archi, activation, sequential, use_bias, mode, use_input, floatx):

    K.set_floatx('float{}'.format(floatx))
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

    IBP=True; forward=True
    if mode=="forward":
        IBP=False
    if mode=="ibp":
        forward=False

    input_tensors= inputs[2:]
    if mode=='ibp':
        input_tensors=[u_c, l_c]
    if mode=='forward':
        input_tensors = [z, W_u, b_u, W_l, b_l]

    _,  output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=True, input_tensors=input_tensors)

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)
    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None]*6
    if mode == 'hybrid':
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    if mode == 'ibp':
        u_c_, l_c_ = f_dense(inputs_[2:])
    if mode =='forward':
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
        decimal=decimal
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)



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
def test_convert_forward_stack_1D(n, archi, activation, sequential, mode, floatx, shared):

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

    _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

    f_dense = Model(inputs[2:], output)
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
    "odd, archi, activation, sequential, mode, floatx, shared",
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
def test_convert_forward_stack_multiD(odd, archi, activation, sequential, mode, floatx, shared):

    use_input = True
    use_bias = True
    shared = True
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

    n_dim = y.shape[1]

    ref_nn_0 = dense_NN_1D(n_dim, archi[:-1], sequential, activation, use_bias)
    ref_nn_1 = dense_NN_1D(archi[-2], archi, sequential, activation, use_bias)
    if sequential:
        ref_nn = Sequential([ref_nn_0, ref_nn_1])
    else:
        ref_nn = Model(inputs[1], ref_nn_1(ref_nn_0(inputs[1])))
    # ref_nn = dense_NN_1D(n_dim, archi, sequential, activation, use_bias)
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

    _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

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
def test_convert_forward_struct0_1D(n, archi, activation, use_input, mode, floatx, shared):

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

    if use_input:
        _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

        f_dense = K.function(inputs[2:], output)
    else:
        input_tmp, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared)
        tmp_model = Model(input_tmp, output)
        f_dense = K.function(inputs[2:], tmp_model(input_tensors))

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
def test_convert_forward_struct0_multiD(odd, archi, activation, use_input, mode, floatx, shared):

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
    input_dim = y.shape[-1]

    ref_nn = toy_struct_v0_1D(input_dim, archi, activation, True)
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

    if use_input:
        _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

        f_dense = K.function(inputs[2:], output)
    else:
        input_tmp, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared)
        tmp_model = Model(input_tmp, output)
        f_dense = K.function(inputs[2:], tmp_model(input_tensors))

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
def test_convert_forward_struct1_1D(n, archi, activation, use_input, mode, floatx, shared):

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

    if use_input:
        _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

        f_dense = K.function(inputs[2:], output)
    else:
        input_tmp, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared)
        tmp_model = Model(input_tmp, output)
        f_dense = K.function(inputs[2:], tmp_model(input_tensors))

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
def test_convert_forward_struct1_multiD(odd, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    input_dim = y.shape[-1]
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_struct_v1_1D(input_dim, archi, True, activation, True)
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

    if use_input:
        _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

        f_dense = K.function(inputs[2:], output)
    else:
        input_tmp, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared)
        tmp_model = Model(input_tmp, output)
        f_dense = K.function(inputs[2:], tmp_model(input_tensors))

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
def test_convert_forward_struct2_1D(n, archi, activation, use_input, mode, floatx, shared):

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

    if use_input:
        _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

        f_dense = K.function(inputs[2:], output)
    else:
        input_tmp, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared)
        tmp_model = Model(input_tmp, output)
        f_dense = K.function(inputs[2:], tmp_model(input_tensors))

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
def test_convert_forward_struct2_multiD(odd, archi, activation, use_input, mode, floatx, shared):

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
    input_dim = y.shape[-1]
    ref_nn = toy_struct_v2_1D(input_dim, archi, True, activation, True)
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

    if use_input:
        _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

        f_dense = K.function(inputs[2:], output)
    else:
        input_tmp, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared)
        tmp_model = Model(input_tmp, output)
        f_dense = K.function(inputs[2:], tmp_model(input_tensors))

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


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, mode, floatx",
    [
        ("channels_last", 0, 0, 1, "hybrid", 32),
        ("channels_last", 0, 0, 1, "forward", 32),
        ("channels_last", 0, 0, 1, "ibp", 32),
        ("channels_last", 0, 0, 1, "hybrid", 64),
        ("channels_last", 0, 0, 1, "forward", 64),
        ("channels_last", 0, 0, 1, "ibp", 64),
        ("channels_last", 0, 0, 1, "hybrid", 16),
        ("channels_last", 0, 0, 1, "forward", 16),
        ("channels_last", 0, 0, 1, "ibp", 16),
        ("channels_first", 0, 0, 1, "hybrid", 32),
        ("channels_first", 0, 0, 1, "forward", 32),
        ("channels_first", 0, 0, 1, "ibp", 32),
        ("channels_first", 0, 0, 1, "hybrid", 64),
        ("channels_first", 0, 0, 1, "forward", 64),
        ("channels_first", 0, 0, 1, "ibp", 64),
        ("channels_first", 0, 0, 1, "hybrid", 16),
        ("channels_first", 0, 0, 1, "forward", 16),
        ("channels_first", 0, 0, 1, "ibp", 16),
    ],
)
def test_convert_conv(data_format, odd, m_0, m_1, mode, floatx):

    use_input = True
    shared = True

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    ref_nn = toy_struct_cnn(dtype=K.floatx())

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

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

    if use_input:
        _, output, _, _ = convert_forward(
            ref_nn, input_dim=z.shape[2], IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors
        )

        f_dense = K.function(inputs[2:], output)
    else:
        input_tmp, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared)
        tmp_model = Model(input_tmp, output)
        f_dense = K.function(inputs[2:], tmp_model(input_tensors))

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
def test_convert_forward_multiple_outputs(odd, archi, activation, use_input, mode, floatx, shared):

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
    input_dim = y.shape[-1]
    ref_nn_0 = toy_struct_v2_1D(input_dim, archi, True, activation, True)
    ref_nn_1 = toy_struct_v1_1D(input_dim, archi, True, activation, True)
    ref_nn = Model(inputs[1], [ref_nn_0(inputs[1]), ref_nn_1(inputs[1])])

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

    f_ref = K.function(inputs, ref_nn(inputs[1]))

    y_ref = f_ref(inputs_)

    if use_input:
        _, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors)

        f_dense = K.function(inputs[2:], output)
    else:
        input_tmp, output, _, _ = convert_forward(ref_nn, IBP=IBP, forward=forward, shared=shared)
        tmp_model = Model(input_tmp, output)
        f_dense = K.function(inputs[2:], tmp_model(input_tensors))

    u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = [None] * 6
    u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = [None] * 6
    if mode == "hybrid":
        z_, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = f_dense(inputs_[2:])
    if mode == "ibp":
        u_c_0, l_c_0, u_c_1, l_c_1 = f_dense(inputs_[2:])
    if mode == "forward":
        z_, w_u_0, b_u_0, w_l_0, b_l_0, _, w_u_1, b_u_1, w_l_1, b_l_1 = f_dense(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref[0],
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_0,
        w_u_0,
        b_u_0,
        l_c_0,
        w_l_0,
        b_l_0,
        "dense_{}".format(odd),
        decimal=decimal,
    )

    assert_output_properties_box(
        x_,
        y_ref[1],
        None,
        None,
        z_[:, 0],
        z_[:, 1],
        u_c_1,
        w_u_1,
        b_u_1,
        l_c_1,
        w_l_1,
        b_l_1,
        "dense_{}".format(odd),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)



# testing with multiples inputs
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
def test_convert_forward_multiple_inputs(odd, archi, activation, use_input, mode, floatx, shared):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs_prev = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_prev_ = get_standard_values_multid_box(odd, dc_decomp=False)
    input_dim = inputs_prev_[1].shape[-1]

    ref_nn_0 = toy_struct_v2_1D(input_dim, archi, True, activation, True)
    ref_nn_1 = toy_struct_v1_1D(input_dim, archi, True, activation, True)

    _, output_0, _, _ = convert_forward(ref_nn_0, IBP=True, forward=True, shared=shared, input_tensors=inputs_prev[2:])
    _, output_1, _, _ = convert_forward(ref_nn_1, IBP=True, forward=True, shared=shared, input_tensors=inputs_prev[2:])

    f_0 = K.function(inputs_prev[2:], output_0)
    f_1 = K.function(inputs_prev[2:], output_1)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = f_0(inputs_prev_[2:])
    inputs_1_ = f_1(inputs_prev_[2:])

    ref_nn = Model([inputs_0[1], inputs_1[1]], Average(dtype=K.floatx())([inputs_0[1], inputs_1[1]]))

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1

    IBP = True
    forward = True

    input_tensors_0 = inputs_0[2:]
    input_tensors_1 = inputs_1[2:]

    if mode == "ibp":
        input_tensors_0 = [u_c_0, l_c_0]
        input_tensors_1 = [u_c_1, l_c_1]
        forward = False
    if mode == "forward":
        input_tensors_0 = [z_0, W_u_0, b_u_0, W_l_0, b_l_0]
        input_tensors_1 = [z_1, W_u_1, b_u_1, W_l_1, b_l_1]
        forward = True
        IBP = False

    _, output, _, _ = convert_forward(
        ref_nn, IBP=IBP, forward=forward, shared=shared, input_tensors=input_tensors_0 + input_tensors_1
    )
    f_dense = K.function(inputs_0[2:] + inputs_1[2:], output)

    y_ref = (ref_nn_0.predict(inputs_prev_[1]) + ref_nn_1.predict(inputs_prev_[1])) / 2.0

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
"""

# network from tutorial 1


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
def test_convert_forward_1D(n, archi, activation, sequential, use_bias, mode, use_input, floatx, helpers):

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
