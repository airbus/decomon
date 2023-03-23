# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from decomon.backward_layers.backward_layers import get_backward
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import DecomonDense, to_decomon
from decomon.utils import Slope


def test_Backward_Dense_1D_box(n, activation, use_bias, previous, mode, floatx, helpers):

    slope = Slope.V_SLOPE

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    layer_ = Dense(1, use_bias=use_bias, activation=activation, dtype=K.floatx())
    input_dim = x.shape[-1]
    layer_(inputs[1])
    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        IBP = True
        forward = True
    elif mode == ForwardMode.IBP:
        IBP = True
        forward = False
    elif mode == ForwardMode.AFFINE:
        IBP = False
        forward = True
    else:
        raise ValueError("Unknown mode.")

    layer = to_decomon(layer_, input_dim, dc_decomp=False, IBP=IBP, forward=forward, shared=True)[0]

    if mode == ForwardMode.HYBRID:
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    elif mode == ForwardMode.AFFINE:
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    elif mode == ForwardMode.IBP:
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output
    else:
        raise ValueError("Unknown mode.")

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1,), dtype=K.floatx())
    # get backward layer
    layer_backward = get_backward(
        layer_, input_dim=input_dim, slope=slope, previous=previous, mode=mode, convex_domain={}
    )

    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode + [w_out, b_out, w_out, b_out])

        f_dense = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])

        output_ = f_dense(inputs_ + [np.ones((len(x), 1, 1)), np.zeros((len(x), 1))])
    else:

        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_dense(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_

    if use_bias:
        W_, bias = layer.get_weights()
    else:
        W_ = layer.get_weights()[0]
        bias = 0.0 * W_[0]

    helpers.assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )
    if use_bias:
        layer.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    else:
        layer.set_weights([2 * np.ones_like(W_)])

    helpers.assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )
    if use_bias:
        layer.set_weights([-2 * np.ones_like(W_), np.ones_like(bias)])
    else:
        layer.set_weights([-2 * np.ones_like(W_)])

    helpers.assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_[:, 0], 0) * W_u + np.minimum(w_u_[:, 0], 0) * W_l, 1)[:, :, None],
        b_u_[:, 0]
        + np.sum(np.maximum(w_u_[:, 0], 0) * b_u[:, :, None], 1)
        + np.sum(np.minimum(w_u_[:, 0], 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_[:, 0], 0) * W_l + np.minimum(w_l_[:, 0], 0) * W_u, 1)[:, :, None],
        b_l_[:, 0]
        + np.sum(np.maximum(w_l_[:, 0], 0) * b_l[:, :, None], 1)
        + np.sum(np.minimum(w_l_[:, 0], 0) * b_u[:, :, None], 1),
        "dense_{}".format(n),
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


def test_Backward_DecomonDense_multiD_box(odd, activation, floatx, mode, previous, helpers):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    input_dim = x.shape[-1]
    layer_ = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())
    layer_(inputs[1])
    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        IBP = True
        forward = True
    elif mode == ForwardMode.IBP:
        IBP = True
        forward = False
    elif mode == ForwardMode.AFFINE:
        IBP = False
        forward = True
    else:
        raise ValueError("Unknown mode.")

    layer = to_decomon(layer_, input_dim, dc_decomp=False, IBP=IBP, forward=forward, shared=True)[0]

    if mode == ForwardMode.HYBRID:
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    elif mode == ForwardMode.AFFINE:
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    elif mode == ForwardMode.IBP:
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output
    else:
        raise ValueError("Unknown mode.")

    # output = layer(inputs[2:])
    # z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1), dtype=K.floatx())
    b_out = Input((1,), dtype=K.floatx())
    # get backward layer
    layer_backward = get_backward(layer_, input_dim=input_dim, previous=previous, mode=mode, convex_domain={})
    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode + [w_out, b_out, w_out, b_out])
        f_dense = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_dense(
            inputs_
            + [
                np.ones((len(x), 1, 1)),
                np.zeros(
                    (
                        len(x),
                        1,
                    )
                ),
            ]
        )
    else:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        # import pdb; pdb.set_trace()
        output_ = f_dense(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_

    helpers.assert_output_properties_box_linear(
        x,
        None,
        z_[:, 0],
        z_[:, 1],
        None,
        np.sum(np.maximum(w_u_, 0) * W_u + np.minimum(w_u_, 0) * W_l, 1)[:, :, None],
        b_u_ + np.sum(np.maximum(w_u_, 0) * b_u[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * b_l[:, :, None], 1),
        None,
        np.sum(np.maximum(w_l_, 0) * W_l + np.minimum(w_l_, 0) * W_u, 1)[:, :, None],
        b_l_ + np.sum(np.maximum(w_l_, 0) * b_l[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * b_u[:, :, None], 1),
        "dense_{}".format(odd),
        decimal=decimal,
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n, activation",
    [(0, "relu")],
)
def test_Backward_DecomonDense_1D_box_model(n, activation, helpers):
    layer = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[2:])
    z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1))
    b_out = Input((1,))
    # get backward layer
    layer_backward = get_backward(layer, input_dim=1)
    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = layer_backward(inputs[2:] + [w_out, b_out, w_out, b_out])

    Model(inputs[2:] + [w_out, b_out], [w_out_u_, b_out_u_, w_out_l_, b_out_l_])
