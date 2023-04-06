# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from decomon.backward_layers.backward_layers import to_backward
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import DecomonActivation, DecomonFlatten
from decomon.layers.decomon_reshape import DecomonReshape
from decomon.utils import Slope


def test_Backward_Activation_1D_box_model(n, activation, mode, floatx, helpers):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonActivation(activation, dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    mode = ForwardMode(mode)
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
    layer_backward = to_backward(layer)
    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = layer_backward(input_mode)
    model = Model(inputs[2:], [w_out_u_, b_out_u_, w_out_l_, b_out_l_])
    w_u_, b_u_, w_l_, b_l_ = model.predict(inputs_[2:])

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
        "activation_{}".format(activation),
        decimal=decimal,
    )

    K.set_floatx("float32")
    K.set_epsilon(eps)


def test_Backward_Activation_1D_box_model_slope(helpers):
    n = 2
    activation = "relu"
    use_bias = False
    mode = ForwardMode.AFFINE

    layer = DecomonActivation(activation, dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]

    outputs_by_slope = {}
    for slope in Slope:
        layer_backward = to_backward(layer, slope=slope, mode=mode, convex_domain={})
        assert layer_backward.slope == slope
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_dense = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        outputs_by_slope[slope] = f_dense(inputs_)

    # check results
    # O_Slope != Z_Slope
    same_outputs_O_n_Z = [
        (a == b).all() for a, b in zip(outputs_by_slope[Slope.O_SLOPE], outputs_by_slope[Slope.Z_SLOPE])
    ]
    assert not all(same_outputs_O_n_Z)

    # V_Slope == Z_Slope
    for a, b in zip(outputs_by_slope[Slope.V_SLOPE], outputs_by_slope[Slope.Z_SLOPE]):
        assert (a == b).all()


def test_Backward_Activation_multiD_box(odd, activation, floatx, mode, helpers):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonActivation(activation, dc_decomp=False, mode=mode, dtype=K.floatx())
    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    mode = ForwardMode(mode)
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
    layer_backward = to_backward(layer)
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
    """
    try:
        assert_output_properties_box_linear(
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
    except ValueError:
        import pdb; pdb.set_trace()
    """
    K.set_floatx("float32")
    K.set_epsilon(eps)


def test_Backward_Flatten_multiD_box(odd, floatx, mode, data_format, helpers):

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonFlatten("channels_last", dc_decomp=False, mode=mode, dtype=K.floatx())
    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    mode = ForwardMode(mode)
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

    w_out = Input((1, 1))
    b_out = Input((1,))
    # get backward layer
    layer_backward = to_backward(
        layer,
    )
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


def test_Backward_Reshape_multiD_box(odd, floatx, mode, data_format, helpers):
    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonReshape((-1,), dc_decomp=False, mode=mode, dtype=K.floatx())
    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    mode = ForwardMode(mode)
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
    layer_backward = to_backward(layer)
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
