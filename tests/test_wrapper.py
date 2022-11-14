from __future__ import absolute_import

import pytest
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential

from decomon import get_lower_box, get_range_box, get_upper_box
from decomon.backward_layers.utils import get_FORWARD, get_IBP
from decomon.models import clone as convert

from . import get_standard_values_multid_box_convert, get_standart_values_1d_box


@pytest.mark.parametrize(
    "n, method, mode, fast",
    [
        (0, "ibp", "ibp", False),
        (1, "ibp", "ibp", False),
        (2, "ibp", "ibp", False),
        (3, "ibp", "ibp", False),
        (4, "ibp", "ibp", False),
        (5, "ibp", "ibp", False),
        (0, "forward", "forward", False),
        (1, "forward", "forward", False),
        (2, "forward", "forward", False),
        (3, "forward", "forward", False),
        (4, "forward", "forward", False),
        (5, "forward", "forward", False),
        (0, "hybrid", "hybrid", False),
        (0, "hybrid", "forward", False),
        (0, "hybrid", "ibp", False),
        (1, "hybrid", "hybrid", False),
        (1, "hybrid", "forward", False),
        (1, "hybrid", "ibp", False),
        (2, "hybrid", "hybrid", False),
        (2, "hybrid", "forward", False),
        (2, "hybrid", "ibp", False),
        (3, "hybrid", "hybrid", False),
        (3, "hybrid", "forward", False),
        (3, "hybrid", "ibp", False),
        (4, "hybrid", "hybrid", False),
        (4, "hybrid", "forward", False),
        (4, "hybrid", "ibp", False),
        (5, "hybrid", "hybrid", False),
        (5, "hybrid", "forward", False),
        (5, "hybrid", "ibp", False),
        (0, "crown-ibp", "ibp", False),
        (1, "crown-ibp", "ibp", False),
        (2, "crown-ibp", "ibp", False),
        (3, "crown-ibp", "ibp", False),
        (4, "crown-ibp", "ibp", False),
        (5, "crown-ibp", "ibp", False),
        (0, "crown-forward", "forward", False),
        (1, "crown-forward", "forward", False),
        (2, "crown-forward", "forward", False),
        (3, "crown-forward", "forward", False),
        (4, "crown-forward", "forward", False),
        (5, "crown-forward", "forward", False),
        (0, "crown-hybrid", "hybrid", False),
        (0, "crown-hybrid", "forward", False),
        (0, "crown-hybrid", "ibp", False),
        (1, "crown-hybrid", "hybrid", False),
        (1, "crown-hybrid", "ibp", False),
        (1, "crown-hybrid", "forward", False),
        (2, "crown-hybrid", "hybrid", False),
        (2, "crown-hybrid", "ibp", False),
        (2, "crown-hybrid", "forward", False),
        (3, "crown-hybrid", "hybrid", False),
        (3, "crown-hybrid", "ibp", False),
        (3, "crown-hybrid", "forward", False),
        (4, "crown-hybrid", "hybrid", False),
        (4, "crown-hybrid", "ibp", False),
        (4, "crown-hybrid", "forward", False),
        (5, "crown-hybrid", "hybrid", False),
        (5, "crown-hybrid", "ibp", False),
        (5, "crown-hybrid", "forward", False),
        (0, "crown", "ibp", False),
        (0, "crown", "hybrid", False),
        (0, "crown", "forward", False),
        (1, "crown", "ibp", False),
        (1, "crown", "hybrid", False),
        (1, "crown", "forward", False),
        (2, "crown", "ibp", False),
        (2, "crown", "hybrid", False),
        (2, "crown", "forward", False),
        (3, "crown", "ibp", False),
        (3, "crown", "hybrid", False),
        (3, "crown", "forward", False),
        (4, "crown", "ibp", False),
        (4, "crown", "hybrid", False),
        (4, "crown", "forward", False),
        (5, "crown", "ibp", False),
        (5, "crown", "hybrid", False),
        (5, "crown", "forward", False),
    ],
)
def test_get_upper_1d_box(n, method, mode, fast):

    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_IBP(mode)
    forward = get_FORWARD(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, forward=forward, mode=mode)

    upper = get_upper_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    try:
        assert (upper - y_ref).min() + 1e-6 >= 0.0
    except AssertionError:
        sequential.save_weights("get_upper_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


@pytest.mark.parametrize(
    "n, method, mode, fast",
    [
        (0, "ibp", "ibp", False),
        (1, "ibp", "ibp", False),
        (2, "ibp", "ibp", False),
        (3, "ibp", "ibp", False),
        (4, "ibp", "ibp", False),
        (5, "ibp", "ibp", False),
        (0, "forward", "forward", False),
        (1, "forward", "forward", False),
        (2, "forward", "forward", False),
        (3, "forward", "forward", False),
        (4, "forward", "forward", False),
        (5, "forward", "forward", False),
        (0, "hybrid", "hybrid", False),
        (0, "hybrid", "forward", False),
        (0, "hybrid", "ibp", False),
        (1, "hybrid", "hybrid", False),
        (1, "hybrid", "forward", False),
        (1, "hybrid", "ibp", False),
        (2, "hybrid", "hybrid", False),
        (2, "hybrid", "forward", False),
        (2, "hybrid", "ibp", False),
        (3, "hybrid", "hybrid", False),
        (3, "hybrid", "forward", False),
        (3, "hybrid", "ibp", False),
        (4, "hybrid", "hybrid", False),
        (4, "hybrid", "forward", False),
        (4, "hybrid", "ibp", False),
        (5, "hybrid", "hybrid", False),
        (5, "hybrid", "forward", False),
        (5, "hybrid", "ibp", False),
        (0, "crown-ibp", "ibp", False),
        (1, "crown-ibp", "ibp", False),
        (2, "crown-ibp", "ibp", False),
        (3, "crown-ibp", "ibp", False),
        (4, "crown-ibp", "ibp", False),
        (5, "crown-ibp", "ibp", False),
        (0, "crown-forward", "forward", False),
        (1, "crown-forward", "forward", False),
        (2, "crown-forward", "forward", False),
        (3, "crown-forward", "forward", False),
        (4, "crown-forward", "forward", False),
        (5, "crown-forward", "forward", False),
        (0, "crown-hybrid", "hybrid", False),
        (0, "crown-hybrid", "forward", False),
        (0, "crown-hybrid", "ibp", False),
        (1, "crown-hybrid", "hybrid", False),
        (1, "crown-hybrid", "ibp", False),
        (1, "crown-hybrid", "forward", False),
        (2, "crown-hybrid", "hybrid", False),
        (2, "crown-hybrid", "ibp", False),
        (2, "crown-hybrid", "forward", False),
        (3, "crown-hybrid", "hybrid", False),
        (3, "crown-hybrid", "ibp", False),
        (3, "crown-hybrid", "forward", False),
        (4, "crown-hybrid", "hybrid", False),
        (4, "crown-hybrid", "ibp", False),
        (4, "crown-hybrid", "forward", False),
        (5, "crown-hybrid", "hybrid", False),
        (5, "crown-hybrid", "ibp", False),
        (5, "crown-hybrid", "forward", False),
        (0, "crown", "ibp", False),
        (0, "crown", "hybrid", False),
        (0, "crown", "forward", False),
        (1, "crown", "ibp", False),
        (1, "crown", "hybrid", False),
        (1, "crown", "forward", False),
        (2, "crown", "ibp", False),
        (2, "crown", "hybrid", False),
        (2, "crown", "forward", False),
        (3, "crown", "ibp", False),
        (3, "crown", "hybrid", False),
        (3, "crown", "forward", False),
        (4, "crown", "ibp", False),
        (4, "crown", "hybrid", False),
        (4, "crown", "forward", False),
        (5, "crown", "ibp", False),
        (5, "crown", "hybrid", False),
        (5, "crown", "forward", False),
    ],
)
def test_get_lower_1d_box(n, method, mode, fast):

    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_IBP(mode)
    forward = get_FORWARD(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, forward=forward, mode=mode)

    lower = get_lower_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    try:
        assert (y_ref - lower).min() + 1e-6 >= 0.0
    except AssertionError:
        sequential.save_weights("get_lower_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


@pytest.mark.parametrize(
    "n, method, mode, fast",
    [
        (0, "ibp", "ibp", False),
        (1, "ibp", "ibp", False),
        (2, "ibp", "ibp", False),
        (3, "ibp", "ibp", False),
        (4, "ibp", "ibp", False),
        (5, "ibp", "ibp", False),
        (0, "forward", "forward", False),
        (1, "forward", "forward", False),
        (2, "forward", "forward", False),
        (3, "forward", "forward", False),
        (4, "forward", "forward", False),
        (5, "forward", "forward", False),
        (0, "hybrid", "hybrid", False),
        (0, "hybrid", "forward", False),
        (0, "hybrid", "ibp", False),
        (1, "hybrid", "hybrid", False),
        (1, "hybrid", "forward", False),
        (1, "hybrid", "ibp", False),
        (2, "hybrid", "hybrid", False),
        (2, "hybrid", "forward", False),
        (2, "hybrid", "ibp", False),
        (3, "hybrid", "hybrid", False),
        (3, "hybrid", "forward", False),
        (3, "hybrid", "ibp", False),
        (4, "hybrid", "hybrid", False),
        (4, "hybrid", "forward", False),
        (4, "hybrid", "ibp", False),
        (5, "hybrid", "hybrid", False),
        (5, "hybrid", "forward", False),
        (5, "hybrid", "ibp", False),
        (0, "crown-ibp", "ibp", False),
        (1, "crown-ibp", "ibp", False),
        (2, "crown-ibp", "ibp", False),
        (3, "crown-ibp", "ibp", False),
        (4, "crown-ibp", "ibp", False),
        (5, "crown-ibp", "ibp", False),
        (0, "crown-forward", "forward", False),
        (1, "crown-forward", "forward", False),
        (2, "crown-forward", "forward", False),
        (3, "crown-forward", "forward", False),
        (4, "crown-forward", "forward", False),
        (5, "crown-forward", "forward", False),
        (0, "crown-hybrid", "hybrid", False),
        (0, "crown-hybrid", "forward", False),
        (0, "crown-hybrid", "ibp", False),
        (1, "crown-hybrid", "hybrid", False),
        (1, "crown-hybrid", "ibp", False),
        (1, "crown-hybrid", "forward", False),
        (2, "crown-hybrid", "hybrid", False),
        (2, "crown-hybrid", "ibp", False),
        (2, "crown-hybrid", "forward", False),
        (3, "crown-hybrid", "hybrid", False),
        (3, "crown-hybrid", "ibp", False),
        (3, "crown-hybrid", "forward", False),
        (4, "crown-hybrid", "hybrid", False),
        (4, "crown-hybrid", "ibp", False),
        (4, "crown-hybrid", "forward", False),
        (5, "crown-hybrid", "hybrid", False),
        (5, "crown-hybrid", "ibp", False),
        (5, "crown-hybrid", "forward", False),
        (0, "crown", "ibp", False),
        (0, "crown", "hybrid", False),
        (0, "crown", "forward", False),
        (1, "crown", "ibp", False),
        (1, "crown", "hybrid", False),
        (1, "crown", "forward", False),
        (2, "crown", "ibp", False),
        (2, "crown", "hybrid", False),
        (2, "crown", "forward", False),
        (3, "crown", "ibp", False),
        (3, "crown", "hybrid", False),
        (3, "crown", "forward", False),
        (4, "crown", "ibp", False),
        (4, "crown", "hybrid", False),
        (4, "crown", "forward", False),
        (5, "crown", "ibp", False),
        (5, "crown", "hybrid", False),
        (5, "crown", "forward", False),
    ],
)
def test_get_range_1d_box(n, method, mode, fast):

    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_IBP(mode)
    forward = get_FORWARD(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, forward=forward, mode=mode)

    upper, lower = get_range_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    try:
        assert (upper - y_ref).min() + 1e-6 >= 0.0
        assert (y_ref - lower).min() + 1e-6 >= 0.0
    except AssertionError:
        sequential.save_weights("get_range_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


@pytest.mark.parametrize(
    "odd, n_subgrad, method, mode, fast",
    [
        (0, 0, "ibp", "ibp", False),
        (1, 0, "ibp", "ibp", False),
        (0, 0, "forward", "forward", False),
        (1, 0, "forward", "forward", False),
        (0, 0, "hybrid", "hybrid", False),
        (0, 0, "hybrid", "forward", False),
        (0, 0, "hybrid", "ibp", False),
        (1, 0, "hybrid", "hybrid", False),
        (1, 0, "hybrid", "ibp", False),
        (1, 0, "hybrid", "forward", False),
        (0, 0, "crown-ibp", "ibp", False),
        (1, 0, "crown-ibp", "ibp", False),
        (0, 0, "crown-forward", "forward", False),
        (1, 0, "crown-forward", "forward", False),
        (0, 0, "crown-hybrid", "hybrid", False),
        (0, 0, "crown-hybrid", "forward", False),
        (0, 0, "crown-hybrid", "ibp", False),
        (1, 0, "crown-hybrid", "hybrid", False),
        (1, 0, "crown-hybrid", "ibp", False),
        (1, 0, "crown-hybrid", "forward", False),
        (0, 0, "crown", "ibp", False),
        (1, 0, "crown", "hybrid", False),
        (1, 0, "crown", "ibp", False),
        (1, 0, "crown", "forward", False),
    ],
)
def test_get_upper_multid_box(odd, n_subgrad, method, mode, fast):

    inputs_ = get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))  #
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_IBP(mode)
    forward = get_FORWARD(mode)
    backward_model = convert(sequential, method="crown", ibp=ibp, forward=forward)
    upper = get_upper_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0


@pytest.mark.parametrize(
    "odd, n_subgrad, method, mode, fast",
    [
        (0, 0, "ibp", "ibp", False),
        (1, 0, "ibp", "ibp", False),
        (0, 0, "forward", "forward", False),
        (1, 0, "forward", "forward", False),
        (0, 0, "hybrid", "hybrid", False),
        (0, 0, "hybrid", "forward", False),
        (0, 0, "hybrid", "ibp", False),
        (1, 0, "hybrid", "hybrid", False),
        (1, 0, "hybrid", "ibp", False),
        (1, 0, "hybrid", "forward", False),
        (0, 0, "crown-ibp", "ibp", False),
        (1, 0, "crown-ibp", "ibp", False),
        (0, 0, "crown-forward", "forward", False),
        (1, 0, "crown-forward", "forward", False),
        (0, 0, "crown-hybrid", "hybrid", False),
        (0, 0, "crown-hybrid", "forward", False),
        (0, 0, "crown-hybrid", "ibp", False),
        (1, 0, "crown-hybrid", "hybrid", False),
        (1, 0, "crown-hybrid", "ibp", False),
        (1, 0, "crown-hybrid", "forward", False),
        (0, 0, "crown", "ibp", False),
        (1, 0, "crown", "hybrid", False),
        (1, 0, "crown", "ibp", False),
        (1, 0, "crown", "forward", False),
    ],
)
def test_get_lower_multid_box(odd, n_subgrad, method, mode, fast):

    inputs_ = get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))  #
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_IBP(mode)
    forward = get_FORWARD(mode)
    backward_model = convert(sequential, method="crown", ibp=ibp, forward=forward)
    lower = get_lower_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    assert (y_ref - lower).min() + 1e-6 >= 0.0


@pytest.mark.parametrize(
    "odd, n_subgrad, method, mode, fast",
    [
        (0, 0, "ibp", "ibp", False),
        (1, 0, "ibp", "ibp", False),
        (0, 0, "forward", "forward", False),
        (1, 0, "forward", "forward", False),
        (0, 0, "hybrid", "hybrid", False),
        (0, 0, "hybrid", "forward", False),
        (0, 0, "hybrid", "ibp", False),
        (1, 0, "hybrid", "hybrid", False),
        (1, 0, "hybrid", "ibp", False),
        (1, 0, "hybrid", "forward", False),
        (0, 0, "crown-ibp", "ibp", False),
        (1, 0, "crown-ibp", "ibp", False),
        (0, 0, "crown-forward", "forward", False),
        (1, 0, "crown-forward", "forward", False),
        (0, 0, "crown-hybrid", "hybrid", False),
        (0, 0, "crown-hybrid", "forward", False),
        (0, 0, "crown-hybrid", "ibp", False),
        (1, 0, "crown-hybrid", "hybrid", False),
        (1, 0, "crown-hybrid", "ibp", False),
        (1, 0, "crown-hybrid", "forward", False),
        (0, 0, "crown", "ibp", False),
        (1, 0, "crown", "hybrid", False),
        (1, 0, "crown", "ibp", False),
        (1, 0, "crown", "forward", False),
    ],
)
def test_get_range_multid_box(odd, n_subgrad, method, mode, fast):

    inputs_ = get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))  #
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_IBP(mode)
    forward = get_FORWARD(mode)
    backward_model = convert(sequential, method="crown", ibp=ibp, forward=forward)
    upper, lower = get_range_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0
    assert (y_ref - lower).min() + 1e-6 >= 0.0
