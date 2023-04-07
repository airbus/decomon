import pytest
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential

from decomon import get_lower_box, get_range_box, get_upper_box
from decomon.backward_layers.utils import get_affine, get_ibp
from decomon.models import clone as convert


def test_get_upper_1d_box(n, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        return

    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, affine=affine, mode=mode)

    upper = get_upper_box(backward_model, z[:, 0], z[:, 1])
    y_ref = sequential.predict(y)

    try:
        assert (upper - y_ref).min() + 1e-6 >= 0.0
    except AssertionError:
        sequential.save_weights("get_upper_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


def test_get_lower_1d_box(n, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        return

    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, affine=affine, mode=mode)

    lower = get_lower_box(backward_model, z[:, 0], z[:, 1])
    y_ref = sequential.predict(y)

    try:
        assert (y_ref - lower).min() + 1e-6 >= 0.0
    except AssertionError:
        sequential.save_weights("get_lower_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


def test_get_range_1d_box(n, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        return

    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, affine=affine, mode=mode)

    upper, lower = get_range_box(backward_model, z[:, 0], z[:, 1])
    y_ref = sequential.predict(y)

    try:
        assert (upper - y_ref).min() + 1e-6 >= 0.0
        assert (y_ref - lower).min() + 1e-6 >= 0.0
    except AssertionError:
        sequential.save_weights("get_range_1d_box_fail_{}_{}_{}.hd5".format(n, method, mode))
        raise AssertionError


def test_get_upper_multid_box(odd, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        return

    inputs_ = helpers.get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))  #
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, affine=affine)
    upper = get_upper_box(backward_model, z[:, 0], z[:, 1])
    y_ref = sequential.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0


def test_get_lower_multid_box(odd, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        return

    inputs_ = helpers.get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))  #
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, affine=affine)
    lower = get_lower_box(backward_model, z[:, 0], z[:, 1])
    y_ref = sequential.predict(y)

    assert (y_ref - lower).min() + 1e-6 >= 0.0


def test_get_range_multid_box(odd, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        return

    inputs_ = helpers.get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="linear", input_dim=y.shape[-1]))  #
    sequential.add(Activation("relu"))
    sequential.add(Dense(1, activation="linear"))
    ibp = get_ibp(mode)
    affine = get_affine(mode)
    backward_model = convert(sequential, method=method, ibp=ibp, affine=affine)
    upper, lower = get_range_box(backward_model, z[:, 0], z[:, 1])
    y_ref = sequential.predict(y)

    assert (upper - y_ref).min() + 1e-6 >= 0.0
    assert (y_ref - lower).min() + 1e-6 >= 0.0
