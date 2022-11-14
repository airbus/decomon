# Test unit for decomon with Dense layers
from __future__ import absolute_import

import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from numpy.testing import assert_almost_equal
from tensorflow.keras.layers import (
    Add,
    Average,
    Concatenate,
    Dot,
    Input,
    Maximum,
    Minimum,
    Multiply,
    Subtract,
)
from tensorflow.keras.models import Model

from decomon.layers.decomon_layers import to_monotonic
from decomon.layers.decomon_merge_layers import (
    DecomonAdd,
    DecomonAverage,
    DecomonConcatenate,
    DecomonDot,
    DecomonMaximum,
    DecomonMinimum,
    DecomonMultiply,
    DecomonSubtract,
)

from . import (
    assert_output_properties_box,
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_standart_values_1d_box,
    get_tensor_decomposition_1d_box,
    get_tensor_decomposition_multid_box,
)

""""
@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_DecomonAdd_1D_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonAdd(dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    y_ = y0_ + y1_
    z_ = z0_
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)



@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_DecomonAverage_1D_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonAverage(dc_decomp=False, mode=mode)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    z_ = z0_
    y_ = (y0_ + y1_) / 2.0
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_DecomonSubstract_1D_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonSubtract(dc_decomp=False, mode=mode, dtype=K.floatx())
    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_
    y_ = y0_ - y1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    z_ = z0_
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_DecomonMaximum_1D_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonMaximum(dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = np.maximum(y0_, y1_)
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_DecomonMinimum_1D_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonMinimum(dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = np.minimum(y0_, y1_)
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_DecomonConcatenate_1D_box(n0, mode, floatx):
    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonConcatenate(axis=-1, dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = np.concatenate([y0_, y1_], -1)
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_epsilon(eps)
    K.set_floatx("float32")
"""


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_DecomonAdd_multiD_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonAdd(dc_decomp=False, mode=mode, dtype=K.floatx())
    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = y0_ + y1_
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )

    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_DecomonSubstract_multiD_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonSubtract(dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = y0_ - y1_
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_DecomonAverage_multiD_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonAverage(dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = (y0_ + y1_) / 2.0
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_DecomonMaximum_multiD_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonMaximum(dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = np.maximum(y0_, y1_)
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_DecomonMinimum_multiD_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonMinimum(dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = np.minimum(y0_, y1_)
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_DecomonConcatenate_multiD_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonConcatenate(dc_decomp=False, mode=mode, axis=-1, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = np.concatenate([y0_, y1_], -1)
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_epsilon(eps)
    K.set_floatx("float32")


### to monotonic


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonAdd_1D_box_to_monotonic(n0):

    ref_op = Add()

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    ref_op([inputs_0[0], inputs_1[0]])

    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)[0]

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = inputs_0_[1] + inputs_1_[1]

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonAverage_1D_box_to_monotonic(n0):

    ref_op = Average()

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)[0]

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = (inputs_0_[1] + inputs_1_[1]) / 2.0

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="average"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonSubtract_1D_box_to_monotonic(n0):

    ref_op = Subtract()

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)[0]

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = inputs_0_[1] - inputs_1_[1]

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="substract"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonMaximum_1D_box_to_monotonic(n0):

    ref_op = Maximum()

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = np.maximum(inputs_0_[1], inputs_1_[1])

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonMinimum_1D_box_to_monotonic(n0):

    ref_op = Minimum()

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = np.minimum(inputs_0_[1], inputs_1_[1])

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonConcatenate_1D_box_to_monotonic(n0):

    ref_op = Concatenate(axis=-1)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = np.concatenate([inputs_0_[1], inputs_1_[1]], -1)

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


#### to_monotonic multiD


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonAdd_multiD_box_to_monotonic(n0):

    ref_op = Add()

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = y0 + y1
    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonSubstract_multiD_box_to_monotonic(n0):

    ref_op = Subtract()

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = y0 - y1

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="substract"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonAverage_multiD_box_to_monotonic(n0):

    ref_op = Average(dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = (y0 + y1) / 2.0

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="average"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonMaximum_multiD_box_to_monotonic(n0):

    ref_op = Maximum(dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = np.maximum(y0, y1)

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonMinimum_multiD_box_to_monotonic(n0):

    ref_op = Minimum(dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = np.minimum(y0, y1)

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="minimum"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonConcatenate_multiD_box_to_monotonic(n0):

    ref_op = Concatenate(axis=-1, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = np.concatenate([y0, y1], -1)

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="concatenate"
    )


# other operators: Multiply, Dot


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (2, "hybrid", 32),
        (3, "hybrid", 32),
        (4, "hybrid", 32),
        (5, "hybrid", 32),
        (6, "hybrid", 32),
        (7, "hybrid", 32),
        (8, "hybrid", 32),
        (9, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (2, "forward", 32),
        (3, "forward", 32),
        (4, "forward", 32),
        (5, "forward", 32),
        (6, "forward", 32),
        (7, "forward", 32),
        (8, "forward", 32),
        (9, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (2, "ibp", 32),
        (3, "ibp", 32),
        (4, "ibp", 32),
        (5, "ibp", 32),
        (6, "ibp", 32),
        (7, "ibp", 32),
        (8, "ibp", 32),
        (9, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (2, "hybrid", 64),
        (3, "hybrid", 64),
        (4, "hybrid", 64),
        (5, "hybrid", 64),
        (6, "hybrid", 64),
        (7, "hybrid", 64),
        (8, "hybrid", 64),
        (9, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (2, "forward", 64),
        (3, "forward", 64),
        (4, "forward", 64),
        (5, "forward", 64),
        (6, "forward", 64),
        (7, "forward", 64),
        (8, "forward", 64),
        (9, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (2, "ibp", 64),
        (3, "ibp", 64),
        (4, "ibp", 64),
        (5, "ibp", 64),
        (6, "ibp", 64),
        (7, "ibp", 64),
        (8, "ibp", 64),
        (9, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (2, "hybrid", 16),
        (3, "hybrid", 16),
        (4, "hybrid", 16),
        (5, "hybrid", 16),
        (6, "hybrid", 16),
        (7, "hybrid", 16),
        (8, "hybrid", 16),
        (9, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (2, "forward", 16),
        (3, "forward", 16),
        (4, "forward", 16),
        (5, "forward", 16),
        (6, "forward", 16),
        (7, "forward", 16),
        (8, "forward", 16),
        (9, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
        (2, "ibp", 16),
        (3, "ibp", 16),
        (4, "ibp", 16),
        (5, "ibp", 16),
        (6, "ibp", 16),
        (7, "ibp", 16),
        (8, "ibp", 16),
        (9, "ibp", 16),
    ],
)
def test_DecomonMultiply_1D_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonMultiply(dc_decomp=False, mode=mode, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = y0_ * y1_
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonMultiply_1D_box_to_monotonic(n0):

    ref_op = Multiply(dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)[0]

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = y0 * y1

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize(
    "n0, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
        (0, "hybrid", 64),
        (1, "hybrid", 64),
        (0, "forward", 64),
        (1, "forward", 64),
        (0, "ibp", 64),
        (1, "ibp", 64),
        (0, "hybrid", 16),
        (1, "hybrid", 16),
        (0, "forward", 16),
        (1, "forward", 16),
        (0, "ibp", 16),
        (1, "ibp", 16),
    ],
)
def test_DecomonMultiply_multiD_box(n0, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    decomon_op = DecomonMultiply(dc_decomp=False, mode=mode, dtype=K.floatx())
    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1

    x0_, y0_, z0_, u_c0_, W_u0_, b_u0_, l_c0_, W_l0_, b_l0_ = inputs_0_
    x1_, y1_, z1_, u_c1_, W_u1_, b_u1_, l_c1_, W_l1_, b_l1_ = inputs_1_

    if mode == "hybrid":
        output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    if mode == "forward":
        output_decomon = decomon_op([z0, W_u0, b_u0, W_l0, b_l0] + [z1, W_u1, b_u1, W_l1, b_l1])
    if mode == "ibp":
        output_decomon = decomon_op([u_c0, l_c0] + [u_c1, l_c1])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)
    y_ = y0_ * y1_
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    u_, w_u_, b_u_, l_, w_l_, b_l_ = [None] * 6
    z_ = z0_
    if mode == "hybrid":
        z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = output_
    if mode == "ibp":
        u_, l_ = output_

    assert_output_properties_box(
        inputs_0_[0], y_, None, None, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add", decimal=decimal
    )
    K.set_floatx("float32")
    K.set_epsilon(eps)


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonMultiply_multiD_box_to_monotonic(n0):

    ref_op = Multiply(dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[1]), dc_decomp=False)[0]

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])
    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = y0 * y1

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


"""
@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonDot_1D_box(n0):

    decomon_op = DecomonDot(axes=(-1, -1), dc_decomp=False, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[2:] + inputs_1[2:])

    model = Model(inputs_0[2:] + inputs_1[2:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[2:] + inputs_1_[2:])
    z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_
    y_ = np.dot(y0, y1)

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonDot_1D_box_to_monotonic(n0):

    ref_op = Dot(axes=(-1, -1), dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])

    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonDot_multiD_box(n0):

    decomon_op = DecomonDot(axes=(-1, -1), dc_decomp=False, dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])
    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])

    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonDot_multiD_box_to_monotonic(n0):

    ref_op = Dot(axes=(1, 1), dtype=K.floatx())

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])
    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])

    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )
"""
