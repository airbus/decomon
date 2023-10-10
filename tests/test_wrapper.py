import numpy as np
import pytest
from keras_core.layers import Activation, Dense
from keras_core.models import Sequential
from numpy.testing import assert_almost_equal

from decomon import get_lower_box, get_range_box, get_upper_box
from decomon.core import get_affine, get_ibp
from decomon.models import clone


@pytest.fixture()
def toy_model_1d():
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=1))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    return sequential


@pytest.fixture()
def toy_model_multid(odd, helpers):
    input_dim = helpers.get_input_dim_multid_box(odd)
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=input_dim))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    return sequential


def test_get_upper_1d_box(toy_model_1d, n, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = clone(toy_model_1d, method=method, ibp=ibp, affine=affine, mode=mode)
    upper = get_upper_box(backward_model, z[:, 0], z[:, 1])
    y_ref = toy_model_1d.predict(y)

    try:
        assert (upper - y_ref).min() + 1e-6 >= 0.0
    except AssertionError:
        toy_model_1d.save_weights("get_upper_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


def test_get_lower_1d_box(toy_model_1d, n, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = clone(toy_model_1d, method=method, final_ibp=ibp, final_affine=affine)
    lower = get_lower_box(backward_model, z[:, 0], z[:, 1])
    y_ref = toy_model_1d.predict(y)

    try:
        assert (y_ref - lower).min() + 1e-6 >= 0.0
    except AssertionError:
        toy_model_1d.save_weights("get_lower_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


def test_get_range_1d_box(toy_model_1d, n, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = clone(toy_model_1d, method=method, final_ibp=ibp, final_affine=affine)
    upper, lower = get_range_box(backward_model, z[:, 0], z[:, 1])
    y_ref = toy_model_1d.predict(y)

    try:
        assert (upper - y_ref).min() + 1e-6 >= 0.0
        assert (y_ref - lower).min() + 1e-6 >= 0.0
    except AssertionError:
        toy_model_1d.save_weights("get_range_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


def test_get_upper_multid_box(toy_model_multid, odd, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    inputs_ = helpers.get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = clone(toy_model_multid, method=method, final_ibp=ibp, final_affine=affine)
    upper = get_upper_box(backward_model, z[:, 0], z[:, 1])
    y_ref = toy_model_multid.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0


def test_get_lower_multid_box(toy_model_multid, odd, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    inputs_ = helpers.get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = clone(toy_model_multid, method=method, final_ibp=ibp, final_affine=affine)
    lower = get_lower_box(backward_model, z[:, 0], z[:, 1])
    y_ref = toy_model_multid.predict(y)

    assert (y_ref - lower).min() + 1e-6 >= 0.0


def test_get_range_multid_box(toy_model_multid, odd, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    inputs_ = helpers.get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = clone(toy_model_multid, method=method, final_ibp=ibp, final_affine=affine)
    upper, lower = get_range_box(backward_model, z[:, 0], z[:, 1])
    y_ref = toy_model_multid.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0
    assert (y_ref - lower).min() + 1e-6 >= 0.0
