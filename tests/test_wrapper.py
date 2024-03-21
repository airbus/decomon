import keras.ops as K
import numpy as np
import pytest
from keras.layers import Activation, Dense, Input
from keras.models import Sequential
from numpy.testing import assert_almost_equal

from decomon import get_adv_box, get_lower_box, get_range_box, get_upper_box
from decomon.models import clone
from decomon.perturbation_domain import BallDomain
from decomon.wrapper import check_adv_box, get_range_noise


@pytest.fixture()
def toy_model_0d():
    sequential = Sequential()
    sequential.add(Input((1,)))
    sequential.add(Dense(1, activation="linear"))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    return sequential


@pytest.fixture()
def toy_model_1d(odd, helpers):
    input_dim = helpers.get_input_dim_1d_box(odd)
    sequential = Sequential()
    sequential.add(Input((input_dim,)))
    sequential.add(Dense(1, activation="linear"))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    return sequential


def test_get_adv_box_0d(toy_model_0d, helpers):
    inputs_ = helpers.get_standard_values_0d_box(n=0)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    score = get_adv_box(toy_model_0d, z[:, 0], z[:, 1], source_labels=0)


def test_check_adv_box_0d(toy_model_0d, helpers):
    inputs_ = helpers.get_standard_values_0d_box(n=0)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    score = check_adv_box(toy_model_0d, z[:, 0], z[:, 1], source_labels=0)


def test_get_upper_0d_box(toy_model_0d, n, method, final_ibp, final_affine, helpers):
    inputs_ = helpers.get_standard_values_0d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    decomon_model = clone(toy_model_0d, method=method, final_ibp=final_ibp, final_affine=final_affine)
    upper = get_upper_box(decomon_model, z[:, 0], z[:, 1])
    y_ref = helpers.predict_on_small_numpy(toy_model_0d, x)

    assert (upper - y_ref).min() + 1e-6 >= 0.0


def test_get_lower_0d_box(toy_model_0d, n, method, final_ibp, final_affine, helpers):
    inputs_ = helpers.get_standard_values_0d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    decomon_model = clone(toy_model_0d, method=method, final_ibp=final_ibp, final_affine=final_affine)
    lower = get_lower_box(decomon_model, z[:, 0], z[:, 1])
    y_ref = helpers.predict_on_small_numpy(toy_model_0d, x)

    assert (y_ref - lower).min() + 1e-6 >= 0.0


def test_get_range_0d_box(toy_model_0d, n, method, final_ibp, final_affine, helpers):
    inputs_ = helpers.get_standard_values_0d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    decomon_model = clone(toy_model_0d, method=method, final_ibp=final_ibp, final_affine=final_affine)
    upper, lower = get_range_box(decomon_model, z[:, 0], z[:, 1])
    y_ref = helpers.predict_on_small_numpy(toy_model_0d, x)

    assert (upper - y_ref).min() + 1e-6 >= 0.0
    assert (y_ref - lower).min() + 1e-6 >= 0.0


def test_get_upper_1d_box(toy_model_1d, odd, method, final_ibp, final_affine, helpers):
    inputs_ = helpers.get_standard_values_1d_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    decomon_model = clone(toy_model_1d, method=method, final_ibp=final_ibp, final_affine=final_affine)
    upper = get_upper_box(decomon_model, z[:, 0], z[:, 1])
    y_ref = helpers.predict_on_small_numpy(toy_model_1d, x)

    assert (upper - y_ref).min() + 1e-6 >= 0.0


def test_get_lower_1d_box(toy_model_1d, odd, method, final_ibp, final_affine, helpers):
    inputs_ = helpers.get_standard_values_1d_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    decomon_model = clone(toy_model_1d, method=method, final_ibp=final_ibp, final_affine=final_affine)
    lower = get_lower_box(decomon_model, z[:, 0], z[:, 1])
    y_ref = helpers.predict_on_small_numpy(toy_model_1d, x)

    assert (y_ref - lower).min() + 1e-6 >= 0.0


def test_get_range_1d_box(toy_model_1d, odd, method, final_ibp, final_affine, helpers):
    inputs_ = helpers.get_standard_values_1d_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    decomon_model = clone(toy_model_1d, method=method, final_ibp=final_ibp, final_affine=final_affine)
    upper, lower = get_range_box(decomon_model, z[:, 0], z[:, 1])
    y_ref = helpers.predict_on_small_numpy(toy_model_1d, x)

    assert (upper - y_ref).min() + 1e-6 >= 0.0
    assert (y_ref - lower).min() + 1e-6 >= 0.0
