# Test unit for decomon with Dense layers
from __future__ import absolute_import

import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from numpy.testing import assert_allclose, assert_almost_equal

from decomon.layers.utils import minus
from decomon.metrics.utils import categorical_cross_entropy

from . import (
    assert_output_properties_box,
    get_standard_values_multid_box,
    get_standart_values_1d_box,
    get_tensor_decomposition_1d_box,
    get_tensor_decomposition_multid_box,
)


@pytest.mark.parametrize(
    "odd, mode, floatx",
    [
        (0, "hybrid", 32),
        (1, "hybrid", 32),
        (0, "forward", 32),
        (1, "forward", 32),
        (0, "ibp", 32),
        (1, "ibp", 32),
    ],
)
def test_categorical_cross_entropy(odd, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)
    if floatx == 16:
        decimal = 2
    else:
        decimal = 5

    inputs_0 = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    if mode == "hybrid":
        output = categorical_cross_entropy(inputs_0[2:], dc_decomp=False, mode=mode)
    if mode == "forward":
        output = categorical_cross_entropy([z_0, W_u_0, b_u_0, W_l_0, b_l_0], dc_decomp=False, mode=mode)
    if mode == "ibp":
        output = categorical_cross_entropy([u_c_0, l_c_0], dc_decomp=False, mode=mode)

    f_ref = K.function(inputs_0, -y_0 + K.log(K.sum(K.exp(y_0), -1))[:, None])
    f_entropy = K.function(inputs_0, output)

    y_ = f_ref(inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_entropy(inputs_)
        assert_output_properties_box(
            x,
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
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_entropy(inputs_)
        assert_output_properties_box(
            x,
            y_,
            None,
            None,
            z_[:, 0],
            z_[:, 1],
            None,
            w_u_,
            b_u_,
            None,
            w_l_,
            b_l_,
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "ibp":
        u_c_, l_c_ = f_entropy(inputs_)
        assert_output_properties_box(
            x,
            y_,
            None,
            None,
            inputs_[2][:, 0],
            inputs_[2][:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
