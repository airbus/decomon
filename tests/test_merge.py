# Test unit for decomon with Dense layers
from __future__ import absolute_import
import pytest
import numpy as np
from decomon.layers.decomon_layers import to_monotonic
from decomon.layers.decomon_merge_layers import (
    DecomonConcatenate,
    DecomonAverage,
    DecomonMaximum,
    DecomonMinimum,
    DecomonAdd,
    DecomonSubtract,
    DecomonMultiply,
    DecomonDot,
)
from tensorflow.keras.layers import Concatenate, Average, Maximum, Minimum, Add, Subtract, Input, Multiply, Dot
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box,
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_tensor_decomposition_multid_box,
)
import tensorflow.python.keras.backend as K
from numpy.testing import assert_almost_equal
from tensorflow.keras.models import Model


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonAdd_1D_box(n0):

    decomon_op = DecomonAdd(dc_decomp=False)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonAverage_1D_box(n0):

    decomon_op = DecomonAverage(dc_decomp=False)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="average"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonSubstract_1D_box(n0):

    decomon_op = DecomonSubtract(dc_decomp=False)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="substract"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonMaximum_1D_box(n0):

    decomon_op = DecomonMaximum(dc_decomp=False)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonMinimum_1D_box(n0):

    decomon_op = DecomonMinimum(dc_decomp=False)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_DecomonConcatenate_1D_box(n0):

    decomon_op = DecomonConcatenate(axis=-1, dc_decomp=False)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonAdd_multiD_box(n0):

    decomon_op = DecomonAdd(dc_decomp=False)

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
def test_DecomonSubstract_multiD_box(n0):

    decomon_op = DecomonSubtract(dc_decomp=False)

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="substract"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonAverage_multiD_box(n0):

    decomon_op = DecomonAverage(dc_decomp=False)

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="average"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonMaximum_multiD_box(n0):

    decomon_op = DecomonMaximum(dc_decomp=False)

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonMinimum_multiD_box(n0):

    decomon_op = DecomonMinimum(dc_decomp=False)

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="minimum"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonConcatenate_multiD_box(n0):

    decomon_op = DecomonConcatenate(axis=-1, dc_decomp=False)

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="concatenate"
    )


### to monotonic


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonAdd_1D_box_to_monotonic(n0):

    ref_op = Add()

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    ref_op([inputs_0[0], inputs_1[0]])

    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

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
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

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
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

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
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

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
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

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
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

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
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[-1]), dc_decomp=False)

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="substract"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonAverage_multiD_box_to_monotonic(n0):

    ref_op = Average()

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="average"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonMaximum_multiD_box_to_monotonic(n0):

    ref_op = Maximum()

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="maximum"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonMinimum_multiD_box_to_monotonic(n0):

    ref_op = Minimum()

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="minimum"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonConcatenate_multiD_box_to_monotonic(n0):

    ref_op = Concatenate(axis=-1)

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
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="concatenate"
    )


# other operators: Multiply, Dot


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonMultiply_1D_box(n0):

    decomon_op = DecomonMultiply(dc_decomp=False)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonMultiply_1D_box_to_monotonic(n0):

    ref_op = Multiply()

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, 1), dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1])
def test_DecomonMultiply_multiD_box(n0):

    decomon_op = DecomonMultiply(dc_decomp=False)

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
def test_DecomonMultiply_multiD_box_to_monotonic(n0):

    ref_op = Multiply()

    inputs_0 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_1 = get_tensor_decomposition_multid_box(n0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    ref_op([inputs_0[0], inputs_1[0]])
    decomon_op = to_monotonic(ref_op, input_dim=(2, x0.shape[1]), dc_decomp=False)

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])
    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonDot_1D_box(n0):

    decomon_op = DecomonDot(axes=(-1, -1), dc_decomp=False)

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n0, dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n0, dc_decomp=False)

    x0, y0, z0, u_c0, W_u0, b_u0, l_c0, W_l0, b_l0 = inputs_0_
    x1, y1, z1, u_c1, W_u1, b_u1, l_c1, W_l1, b_l1 = inputs_1_

    output_decomon = decomon_op(inputs_0[1:] + inputs_1[1:])

    model = Model(inputs_0[1:] + inputs_1[1:], output_decomon)

    # output_ = K.function(inputs_0[1:]+inputs_1[1:], output_decomon)(inputs_0_[1:]+inputs_1_[1:])
    output_ = model.predict(inputs_0_[1:] + inputs_1_[1:])
    y_, z_, u_, w_u_, b_u_, l_, w_l_, b_l_ = output_

    assert_output_properties_box_linear(
        inputs_0_[0], y_, z_[:, 0], z_[:, 1], u_, w_u_, b_u_, l_, w_l_, b_l_, name="add"
    )


@pytest.mark.parametrize("n0", [0, 1, 2, 3, 4, 5, 6, 7])
def test_DecomonDot_1D_box_to_monotonic(n0):

    ref_op = Dot(axes=(-1, -1))

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

    decomon_op = DecomonDot(axes=(-1, -1), dc_decomp=False)

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

    ref_op = Dot(axes=(1, 1))

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
