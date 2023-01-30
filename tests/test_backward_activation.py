import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Input

from decomon.backward_layers.activations import backward_relu, backward_softsign
from decomon.layers.activations import softsign
from decomon.utils import relu_


@pytest.mark.parametrize(
    "activation_func, tensor_func, funcname",
    [
        (relu_, backward_relu, "relu"),
        (softsign, backward_softsign, "softsign"),
    ],
)
def test_activation_backward_1D_box(n, mode, previous, floatx, activation_func, tensor_func, funcname, helpers):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standart_values_1d_box(n, dc_decomp=False)

    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs

    x_, y_, z_, u_c_, W_u_, B_u_, l_c_, W_l_, B_l_ = inputs_

    if mode == "hybrid":
        input_mode = inputs[2:]
        output = activation_func(input_mode, dc_decomp=False, mode=mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode == "forward":
        input_mode = [z, W_u, b_u, W_l, b_l]
        output = activation_func(input_mode, dc_decomp=False, mode=mode)
        z_0, _, _, _, _ = output
    if mode == "ibp":
        input_mode = [u_c, l_c]
        output = activation_func(input_mode, dc_decomp=False, mode=mode)
        z_0 = z_
        u_c_0, l_c_0 = output

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1), dtype=K.floatx())

    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = tensor_func(
            input_mode + [w_out, b_out, w_out, b_out], mode=mode, previous=previous
        )
        f_func = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_func(inputs_ + [np.ones((len(x_), 1, 1)), np.zeros((len(x_), 1))])
    else:
        w_out_u, b_out_u, w_out_l, b_out_l = tensor_func(input_mode, mode=mode, previous=previous)
        f_func = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_func(inputs_)

    w_u_, b_u_, w_l_, b_l_ = output_
    w_u_b = np.sum(np.maximum(w_u_, 0) * W_u_ + np.minimum(w_u_, 0) * W_l_, 1)[:, :, None]
    b_u_b = b_u_ + np.sum(np.maximum(w_u_, 0) * B_u_[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * B_l_[:, :, None], 1)
    w_l_b = np.sum(np.maximum(w_l_, 0) * W_l_ + np.minimum(w_l_, 0) * W_u_, 1)[:, :, None]
    b_l_b = b_l_ + np.sum(np.maximum(w_l_, 0) * B_l_[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * B_u_[:, :, None], 1)

    helpers.assert_output_properties_box_linear(
        x_, None, z_[:, 0], z_[:, 1], None, w_u_b, b_u_b, None, w_l_b, b_l_b, f"{funcname}_{n}", decimal=decimal
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")
