# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from decomon.backward_layers.backward_layers import to_backward
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import DecomonDense
from decomon.utils import Slope


def test_Backward_Dense_1D_box(n, use_bias, mode, floatx, helpers):

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

    layer_ = Dense(1, use_bias=use_bias, dtype=K.floatx())
    input_dim = x.shape[-1]
    layer_(inputs[1])
    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        input_mode = inputs[2:]
    elif mode == ForwardMode.AFFINE:
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
    elif mode == ForwardMode.IBP:
        input_mode = [inputs[3], inputs[6]]
    else:
        raise ValueError("Unknown mode.")

    # get backward layer
    layer_backward = to_backward(layer_, input_dim=input_dim, slope=slope, mode=mode, convex_domain={})

    w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
    f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
    output_ = f_dense(inputs_)
    w_u_, b_u_, w_l_, b_l_ = output_

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


def test_Backward_DecomonDense_multiD_box(odd, floatx, mode, helpers):
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
    layer_ = Dense(1, use_bias=True, dtype=K.floatx())
    layer_(inputs[1])
    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        input_mode = inputs[2:]
    elif mode == ForwardMode.AFFINE:
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
    elif mode == ForwardMode.IBP:
        input_mode = [inputs[3], inputs[6]]
    else:
        raise ValueError("Unknown mode.")

    # get backward layer
    layer_backward = to_backward(layer_, input_dim=input_dim, mode=mode, convex_domain={})
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


def test_Backward_DecomonDense_1D_box_model(n, helpers):
    layer = DecomonDense(1, use_bias=True, dc_decomp=False)

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = layer(inputs[2:])
    z_0, u_c_0, _, _, l_c_0, _, _ = output

    w_out = Input((1, 1))
    b_out = Input((1,))
    # get backward layer
    layer_backward = to_backward(layer, input_dim=1)
    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = layer_backward(inputs[2:] + [w_out, b_out, w_out, b_out])

    Model(inputs[2:] + [w_out, b_out], [w_out_u_, b_out_u_, w_out_l_, b_out_l_])
