# Test unit for decomon with Dense layers


import pytest
import tensorflow.python.keras.backend as K

from decomon.layers.activations import sigmoid, softmax, softsign, tanh


@pytest.mark.parametrize(
    "activation_func, tensor_func, funcname, decimal",
    [
        (sigmoid, K.sigmoid, "sigmoid", 5),
        (tanh, K.tanh, "tanh", 5),
        (softsign, K.softsign, "softsign", 4),
        (softmax, K.softmax, "softmax", 4),
    ],
)
@pytest.mark.parametrize("n,mode,floatx", [(0, "forward", 32)])
def test_activation_1D_box(n, mode, floatx, helpers, activation_func, tensor_func, funcname, decimal):
    # softmax: test only n=0,3
    if funcname == "softmax":
        if n not in {0, 3}:
            return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-4)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs  # tensors
    (
        x_0,
        y_0,
        z_0,
        u_c_0,
        W_u_0,
        b_u_0,
        l_c_0,
        W_l_0,
        b_l_0,
    ) = inputs_  # numpy values

    if mode == "hybrid":
        output = activation_func(inputs[2:], dc_decomp=False, mode=mode)
    if mode == "forward":
        output = activation_func([z, W_u, b_u, W_l, b_l], dc_decomp=False, mode=mode)
    if mode == "ibp":
        output = activation_func([u_c, l_c], dc_decomp=False, mode=mode)

    f_func = K.function(inputs[2:], output)
    f_ref = K.function(inputs, tensor_func(y))
    y_ = f_ref(inputs_)
    z_ = z_0
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_func(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_func(inputs_[2:])
        u_c_, l_c_ = [None] * 2
    if mode == "ibp":
        u_c_, l_c_ = f_func(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    helpers.assert_output_properties_box(
        x_0,
        y_,
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
        f"{funcname}_{n}",
        decimal=decimal,
    )

    K.set_floatx("float32")
    K.set_epsilon(eps)
