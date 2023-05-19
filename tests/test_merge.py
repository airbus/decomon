# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Add,
    Average,
    Concatenate,
    Maximum,
    Minimum,
    Multiply,
    Subtract,
)
from tensorflow.keras.models import Model

from decomon.layers.convert import to_decomon
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_merge_layers import (
    DecomonAdd,
    DecomonAverage,
    DecomonConcatenate,
    DecomonMaximum,
    DecomonMinimum,
    DecomonMultiply,
    DecomonSubtract,
)


def add_op(x, y):
    return x + y


def subtract_op(x, y):
    return x - y


def multiply_op(x, y):
    return x * y


def average_op(x, y):
    return (x + y) / 2.0


def concatenate_op(x, y):
    return np.concatenate([x, y], -1)


@pytest.mark.parametrize(
    "decomon_op_class, tensor_op, decomon_op_kwargs",
    [
        (DecomonAdd, add_op, {}),
        (DecomonSubtract, subtract_op, {}),
        (DecomonAverage, average_op, {}),
        (DecomonMaximum, np.maximum, {}),
        (DecomonMinimum, np.minimum, {}),
        (DecomonConcatenate, concatenate_op, {"axis": -1}),
        (DecomonMultiply, multiply_op, {}),
    ],
)
def test_DecomonOp_1D_box(decomon_op_class, tensor_op, decomon_op_kwargs, n, mode, floatx, helpers):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = decomon_op_class(dc_decomp=False, mode=mode, dtype=K.floatx(), **decomon_op_kwargs)

    inputs_0 = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    inputs_1_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    elif mode == ForwardMode.AFFINE:
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    elif mode == ForwardMode.IBP:
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])
    else:
        raise ValueError("Unknown mode.")

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = tensor_op(y0_, y1_)
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == ForwardMode.HYBRID:
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    elif mode == ForwardMode.IBP:
        u_, l_ = output_
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "decomon_op_class, tensor_op, decomon_op_kwargs",
    [
        (DecomonAdd, add_op, {}),
        (DecomonSubtract, subtract_op, {}),
        (DecomonAverage, average_op, {}),
        (DecomonMaximum, np.maximum, {}),
        (DecomonMinimum, np.minimum, {}),
        (DecomonConcatenate, concatenate_op, {"axis": -1}),
        (DecomonMultiply, multiply_op, {}),
    ],
)
def test_DecomonOp_multiD_box(decomon_op_class, tensor_op, decomon_op_kwargs, odd, mode, floatx, helpers):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = decomon_op_class(dc_decomp=False, mode=mode, dtype=K.floatx(), **decomon_op_kwargs)

    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_0_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    inputs_1_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    elif mode == ForwardMode.AFFINE:
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    elif mode == ForwardMode.IBP:
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])
    else:
        raise ValueError("Unknown mode.")

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = tensor_op(y0_, y1_)
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == ForwardMode.HYBRID:
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    elif mode == ForwardMode.IBP:
        u_, l_ = output_
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


### to decomon


@pytest.mark.parametrize(
    "layer_class, tensor_op, layer_kwargs",
    [
        (Add, add_op, {}),
        (Average, average_op, {}),
        (Subtract, subtract_op, {}),
        (Maximum, np.maximum, {}),
        (Minimum, np.minimum, {}),
        (Concatenate, concatenate_op, {"axis": -1}),
        (Multiply, multiply_op, {}),
    ],
)
def test_Decomon_1D_box_to_decomon(layer_class, tensor_op, layer_kwargs, n, helpers):

    ref_op = layer_class(dtype=K.floatx(), **layer_kwargs)

    inputs_0 = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    inputs_1_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)

    ref_op([inputs_0[0], inputs_1[0]])

    decomon_op = to_decomon(ref_op, input_dim=1, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = tensor_op(inputs_0_[1], inputs_1_[1])

    helpers.assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


#### to_decomon multiD


@pytest.mark.parametrize(
    "layer_class, tensor_op, layer_kwargs",
    [
        (Add, add_op, {}),
        (Average, average_op, {}),
        (Subtract, subtract_op, {}),
        (Maximum, np.maximum, {}),
        (Minimum, np.minimum, {}),
        (Concatenate, concatenate_op, {"axis": -1}),
        (Multiply, multiply_op, {}),
    ],
)
def test_Decomon_multiD_box_to_decomon(layer_class, tensor_op, layer_kwargs, odd, helpers):

    ref_op = layer_class(dtype=K.floatx(), **layer_kwargs)

    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_1 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_0_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    inputs_1_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_decomon(ref_op, input_dim=x0.shape[-1], dc_decomp=False)

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = tensor_op(y0, y1)
    helpers.assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )
