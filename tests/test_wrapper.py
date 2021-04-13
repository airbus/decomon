from __future__ import absolute_import
import pytest
from decomon.models.decomon_sequential import convert

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from . import (
    get_standart_values_1d_box,
    get_standard_values_multid_box_convert,
)
from decomon import get_upper_box, get_lower_box, get_range_box


@pytest.mark.parametrize(
    "odd, n_subgrad, mode, fast",
    [
        (0, 0, "forward", False),
        (1, 0, "forward", False),
        (0, 1, "forward", False),
        (1, 1, "forward", False),
        (0, 5, "forward", False),
        (1, 5, "forward", False),
        (0, 0, "forward", True),
        (1, 0, "forward", True),
        (0, 1, "forward", True),
        (1, 1, "forward", True),
        (0, 5, "forward", True),
        (1, 5, "forward", True),
        (0, 0, "backward", False),
        (1, 0, "backward", False),
        (0, 1, "backward", False),
        (1, 1, "backward", False),
        (0, 5, "backward", False),
        (1, 5, "backward", False),
        (0, 0, "backward", True),
        (1, 0, "backward", True),
        (0, 1, "backward", True),
        (1, 1, "backward", True),
        (0, 5, "backward", True),
        (1, 5, "backward", True),
    ],
)
def test_get_upper_multid_box(odd, n_subgrad, mode, fast):

    inputs_ = get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    backward_model = convert(sequential, dc_decomp=False, n_subgrad=n_subgrad, mode=mode)

    upper = get_upper_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    try:
        assert (upper - y_ref).min() + 1e-6 >= 0.0
    except AssertionError:
        import pdb

        pdb.set_trace()


@pytest.mark.parametrize(
    "odd, n_subgrad, mode, fast",
    [
        (0, 0, "forward", False),
        (1, 0, "forward", False),
        (0, 1, "forward", False),
        (1, 1, "forward", False),
        (0, 5, "forward", False),
        (1, 5, "forward", False),
        (0, 0, "forward", True),
        (1, 0, "forward", True),
        (0, 1, "forward", True),
        (1, 1, "forward", True),
        (0, 5, "forward", True),
        (1, 5, "forward", True),
        (0, 0, "backward", False),
        (1, 0, "backward", False),
        (0, 1, "backward", False),
        (1, 1, "backward", False),
        (0, 5, "backward", False),
        (1, 5, "backward", False),
        (0, 0, "backward", True),
        (1, 0, "backward", True),
        (0, 1, "backward", True),
        (1, 1, "backward", True),
        (0, 5, "backward", True),
        (1, 5, "backward", True),
    ],
)
def test_get_lower_multid_box(odd, n_subgrad, mode, fast):

    inputs_ = get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    backward_model = convert(sequential, dc_decomp=False, n_subgrad=n_subgrad, mode=mode)

    lower = get_lower_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    assert (lower - y_ref).max() - 1e-6 <= 0.0


@pytest.mark.parametrize(
    "odd, n_subgrad, mode, fast",
    [
        (0, 0, "forward", False),
        (1, 0, "forward", False),
        (0, 1, "forward", False),
        (1, 1, "forward", False),
        (0, 5, "forward", False),
        (1, 5, "forward", False),
        (0, 0, "forward", True),
        (1, 0, "forward", True),
        (0, 1, "forward", True),
        (1, 1, "forward", True),
        (0, 5, "forward", True),
        (1, 5, "forward", True),
        (0, 0, "backward", False),
        (1, 0, "backward", False),
        (0, 1, "backward", False),
        (1, 1, "backward", False),
        (0, 5, "backward", False),
        (1, 5, "backward", False),
        (0, 0, "backward", True),
        (1, 0, "backward", True),
        (0, 1, "backward", True),
        (1, 1, "backward", True),
        (0, 5, "backward", True),
        (1, 5, "backward", True),
    ],
)
def test_get_range_multid_box(odd, n_subgrad, mode, fast):

    inputs_ = get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    backward_model = convert(sequential, dc_decomp=False, n_subgrad=n_subgrad, mode=mode)

    upper, lower = get_range_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0
    assert (lower - y_ref).max() - 1e-6 <= 0.0


@pytest.mark.parametrize(
    "n, n_subgrad, mode, fast",
    [
        (0, 0, "forward", False),
        (1, 0, "forward", False),
        (2, 0, "forward", False),
        (3, 0, "forward", False),
        (4, 0, "forward", False),
        (5, 0, "forward", False),
        (0, 1, "forward", False),
        (1, 1, "forward", False),
        (2, 1, "forward", False),
        (3, 1, "forward", False),
        (4, 1, "forward", False),
        (5, 1, "forward", False),
        (0, 5, "forward", False),
        (1, 5, "forward", False),
        (2, 5, "forward", False),
        (3, 5, "forward", False),
        (4, 5, "forward", False),
        (5, 5, "forward", False),
        (0, 0, "forward", True),
        (1, 0, "forward", True),
        (2, 0, "forward", True),
        (3, 0, "forward", True),
        (4, 0, "forward", True),
        (5, 0, "forward", True),
        (0, 1, "forward", True),
        (1, 1, "forward", True),
        (2, 1, "forward", True),
        (3, 1, "forward", True),
        (4, 1, "forward", True),
        (5, 1, "forward", True),
        (0, 5, "forward", True),
        (1, 5, "forward", True),
        (2, 5, "forward", True),
        (3, 5, "forward", True),
        (4, 5, "forward", True),
        (5, 5, "forward", True),
        (0, 0, "backward", False),
        (1, 0, "backward", False),
        (2, 0, "backward", False),
        (3, 0, "backward", False),
        (4, 0, "backward", False),
        (5, 0, "backward", False),
        (0, 1, "backward", False),
        (1, 1, "backward", False),
        (2, 1, "backward", False),
        (3, 1, "backward", False),
        (4, 1, "backward", False),
        (5, 1, "backward", False),
        (0, 5, "backward", False),
        (1, 5, "backward", False),
        (2, 5, "backward", False),
        (3, 5, "backward", False),
        (4, 5, "backward", False),
        (5, 5, "backward", False),
        (0, 0, "backward", True),
        (1, 0, "backward", True),
        (2, 0, "backward", True),
        (3, 0, "backward", True),
        (4, 0, "backward", True),
        (5, 0, "backward", True),
        (0, 1, "backward", True),
        (1, 1, "backward", True),
        (2, 1, "backward", True),
        (3, 1, "backward", True),
        (4, 1, "backward", True),
        (5, 1, "backward", True),
        (0, 5, "backward", True),
        (1, 5, "backward", True),
        (2, 5, "backward", True),
        (3, 5, "backward", True),
        (4, 5, "backward", True),
        (5, 5, "backward", True),
    ],
)
def test_get_upper_1d_box(n, n_subgrad, mode, fast):

    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    backward_model = convert(sequential, dc_decomp=False, n_subgrad=n_subgrad, mode=mode)

    upper = get_upper_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0


@pytest.mark.parametrize(
    "n, n_subgrad, mode, fast",
    [
        (0, 0, "forward", False),
        (1, 0, "forward", False),
        (2, 0, "forward", False),
        (3, 0, "forward", False),
        (4, 0, "forward", False),
        (5, 0, "forward", False),
        (0, 1, "forward", False),
        (1, 1, "forward", False),
        (2, 1, "forward", False),
        (3, 1, "forward", False),
        (4, 1, "forward", False),
        (5, 1, "forward", False),
        (0, 5, "forward", False),
        (1, 5, "forward", False),
        (2, 5, "forward", False),
        (3, 5, "forward", False),
        (4, 5, "forward", False),
        (5, 5, "forward", False),
        (0, 0, "forward", True),
        (1, 0, "forward", True),
        (2, 0, "forward", True),
        (3, 0, "forward", True),
        (4, 0, "forward", True),
        (5, 0, "forward", True),
        (0, 1, "forward", True),
        (1, 1, "forward", True),
        (2, 1, "forward", True),
        (3, 1, "forward", True),
        (4, 1, "forward", True),
        (5, 1, "forward", True),
        (0, 5, "forward", True),
        (1, 5, "forward", True),
        (2, 5, "forward", True),
        (3, 5, "forward", True),
        (4, 5, "forward", True),
        (5, 5, "forward", True),
        (0, 0, "backward", False),
        (1, 0, "backward", False),
        (2, 0, "backward", False),
        (3, 0, "backward", False),
        (4, 0, "backward", False),
        (5, 0, "backward", False),
        (0, 1, "backward", False),
        (1, 1, "backward", False),
        (2, 1, "backward", False),
        (3, 1, "backward", False),
        (4, 1, "backward", False),
        (5, 1, "backward", False),
        (0, 5, "backward", False),
        (1, 5, "backward", False),
        (2, 5, "backward", False),
        (3, 5, "backward", False),
        (4, 5, "backward", False),
        (5, 5, "backward", False),
        (0, 0, "backward", True),
        (1, 0, "backward", True),
        (2, 0, "backward", True),
        (3, 0, "backward", True),
        (4, 0, "backward", True),
        (5, 0, "backward", True),
        (0, 1, "backward", True),
        (1, 1, "backward", True),
        (2, 1, "backward", True),
        (3, 1, "backward", True),
        (4, 1, "backward", True),
        (5, 1, "backward", True),
        (0, 5, "backward", True),
        (1, 5, "backward", True),
        (2, 5, "backward", True),
        (3, 5, "backward", True),
        (4, 5, "backward", True),
        (5, 5, "backward", True),
    ],
)
def test_get_lower_1d_box(n, n_subgrad, mode, fast):

    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    backward_model = convert(sequential, dc_decomp=False, n_subgrad=n_subgrad, mode=mode)

    lower = get_lower_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    assert (lower - y_ref).max() - 1e-6 <= 0.0


@pytest.mark.parametrize(
    "n, n_subgrad, mode, fast",
    [
        (0, 0, "forward", False),
        (1, 0, "forward", False),
        (2, 0, "forward", False),
        (3, 0, "forward", False),
        (4, 0, "forward", False),
        (5, 0, "forward", False),
        (0, 1, "forward", False),
        (1, 1, "forward", False),
        (2, 1, "forward", False),
        (3, 1, "forward", False),
        (4, 1, "forward", False),
        (5, 1, "forward", False),
        (0, 5, "forward", False),
        (1, 5, "forward", False),
        (2, 5, "forward", False),
        (3, 5, "forward", False),
        (4, 5, "forward", False),
        (5, 5, "forward", False),
        (0, 0, "forward", True),
        (1, 0, "forward", True),
        (2, 0, "forward", True),
        (3, 0, "forward", True),
        (4, 0, "forward", True),
        (5, 0, "forward", True),
        (0, 1, "forward", True),
        (1, 1, "forward", True),
        (2, 1, "forward", True),
        (3, 1, "forward", True),
        (4, 1, "forward", True),
        (5, 1, "forward", True),
        (0, 5, "forward", True),
        (1, 5, "forward", True),
        (2, 5, "forward", True),
        (3, 5, "forward", True),
        (4, 5, "forward", True),
        (5, 5, "forward", True),
        (0, 0, "backward", False),
        (1, 0, "backward", False),
        (2, 0, "backward", False),
        (3, 0, "backward", False),
        (4, 0, "backward", False),
        (5, 0, "backward", False),
        (0, 1, "backward", False),
        (1, 1, "backward", False),
        (2, 1, "backward", False),
        (3, 1, "backward", False),
        (4, 1, "backward", False),
        (5, 1, "backward", False),
        (0, 5, "backward", False),
        (1, 5, "backward", False),
        (2, 5, "backward", False),
        (3, 5, "backward", False),
        (4, 5, "backward", False),
        (5, 5, "backward", False),
        (0, 0, "backward", True),
        (1, 0, "backward", True),
        (2, 0, "backward", True),
        (3, 0, "backward", True),
        (4, 0, "backward", True),
        (5, 0, "backward", True),
        (0, 1, "backward", True),
        (1, 1, "backward", True),
        (2, 1, "backward", True),
        (3, 1, "backward", True),
        (4, 1, "backward", True),
        (5, 1, "backward", True),
        (0, 5, "backward", True),
        (1, 5, "backward", True),
        (2, 5, "backward", True),
        (3, 5, "backward", True),
        (4, 5, "backward", True),
        (5, 5, "backward", True),
    ],
)
def test_get_range_1d_box(n, n_subgrad, mode, fast):

    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    backward_model = convert(sequential, dc_decomp=False, n_subgrad=n_subgrad, mode=mode)

    upper, lower = get_range_box(backward_model, z[:, 0], z[:, 1], fast=fast)
    y_ref = sequential.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0
    assert (lower - y_ref).max() - 1e-6 <= 0.0
