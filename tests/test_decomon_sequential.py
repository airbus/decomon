from __future__ import absolute_import
import pytest
from decomon.models.decomon_sequential import (
    clone_sequential_model,
    convert,
    DecomonModel,
)

# from ..models.monotonic_sequential import clone_sequential_model, clone_functional_model
from numpy.testing import assert_almost_equal
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box,
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_tensor_decomposition_multid_box,
)
import tensorflow.python.keras.backend as K


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_convert_model_1d_box(n):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    monotonic_model = convert(sequential, dc_decomp=True)

    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)
    x, _, z, _, _, _, _, _, _, h, g = inputs
    x_i, _, z_i, _, _, _, _, _, _, h_i, g_i = inputs_

    output_ref = sequential(inputs[0])
    f_ref = K.function(inputs, output_ref)

    output = monotonic_model([x, z, h, g])

    f_clone = K.function([x, z, h, g], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_clone([x_i, z_i, h_i, g_i])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box(
        x_i,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "clone_sequential_{}".format(n),
        decimal=5,
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_convert_model_1d_nodc(n):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    monotonic_model = convert(sequential)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, _, z, _, _, _, _, _, _ = inputs
    x_i, _, z_i, _, _, _, _, _, _ = inputs_

    output_ref = sequential(inputs[0])
    f_ref = K.function(inputs, output_ref)

    output = monotonic_model([x, z])

    f_clone = K.function([x, z], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_clone([x_i, z_i])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, 'nodc')


# testing that the cloning function is working
# assuming that the function to_monotonic is working correctly and has been tested throughly
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_clone_sequential_model_1d_box(n):
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(sequential, input_dim=1, dc_decomp=True)

    assert isinstance(monotonic_model, DecomonModel)
    assert not isinstance(sequential, DecomonModel)

    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = monotonic_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "clone_sequential_{}".format(n),
        decimal=5,
    )


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_clone_sequential_model_1d_box_nodc(n):
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(sequential, input_dim=1, dc_decomp=False)

    assert isinstance(monotonic_model, DecomonModel)
    assert not isinstance(sequential, DecomonModel)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = monotonic_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_,
                                        b_u_, l_c_, w_l_, b_l_, 'nodc')


@pytest.mark.parametrize("odd", [0, 1])
def test_clone_sequential_model_multid_box(odd):

    inputs = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_
    input_dim = x.shape[-1]

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(sequential, input_dim=input_dim, dc_decomp=True)

    assert isinstance(monotonic_model, DecomonModel)
    assert not isinstance(sequential, DecomonModel)

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = monotonic_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=6, err_msg="reconstruction error")
    assert_output_properties_box(
        x,
        y_,
        h_,
        g_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "clone_sequential_{}".format(odd),
        decimal=5,
    )


@pytest.mark.parametrize("odd", [0, 1])
def test_clone_sequential_model_multid_box_nodc(odd):

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    input_dim = x.shape[-1]

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(sequential, input_dim=input_dim)

    assert isinstance(monotonic_model, DecomonModel)
    assert not isinstance(sequential, DecomonModel)

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = monotonic_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_,
                                        b_u_, l_c_, w_l_, b_l_, 'nodc')
