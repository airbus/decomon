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
    get_standard_values_multid_box_convert,
)
import tensorflow.python.keras.backend as K


@pytest.mark.parametrize(
    "n, n_subgrad",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
    ],
)
def test_convert_model_1d_box(n, n_subgrad):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    monotonic_model = convert(sequential, dc_decomp=True, n_subgrad=n_subgrad, IBP=True, forward=True)

    for layer in monotonic_model.layers:
        if hasattr(layer, "n_subgrad"):
            assert layer.n_subgrad == n_subgrad

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


@pytest.mark.parametrize(
    "n, n_subgrad",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
    ],
)
def test_convert_model_1d_nodc(n, n_subgrad):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    monotonic_model = convert(sequential, n_subgrad=n_subgrad, IBP=True, forward=True)

    for layer in monotonic_model.layers:
        if hasattr(layer, "n_subgrad"):
            assert layer.n_subgrad == n_subgrad

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
    assert_output_properties_box_linear(x_i, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")


# testing that the cloning function is working
# assuming that the function to_monotonic is working correctly and has been tested throughly
@pytest.mark.parametrize(
    "n, n_subgrad",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
        (9, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
        (6, 5),
        (7, 5),
        (8, 5),
        (9, 5),
    ],
)
def test_clone_sequential_model_1d_box(n, n_subgrad):
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(
        sequential, input_dim=1, dc_decomp=True, n_subgrad=n_subgrad, IBP=True, forward=True
    )

    for layer in monotonic_model.layers:
        if hasattr(layer, "n_subgrad"):
            assert layer.n_subgrad == n_subgrad

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


@pytest.mark.parametrize(
    "n, n_subgrad",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
        (9, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
        (6, 5),
        (7, 5),
        (8, 5),
        (9, 5),
    ],
)
def test_clone_sequential_model_1d_box_nodc(n, n_subgrad):
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(
        sequential, input_dim=1, dc_decomp=False, n_subgrad=n_subgrad, IBP=True, forward=True
    )

    for layer in monotonic_model.layers:
        if hasattr(layer, "n_subgrad"):
            assert layer.n_subgrad == n_subgrad

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
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")


@pytest.mark.parametrize("odd, n_subgrad", [(0, 0), (1, 0), (0, 1), (1, 1), (0, 5), (1, 5)])
def test_clone_sequential_model_multid_box(odd, n_subgrad):

    inputs = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_
    input_dim = x.shape[-1]

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(
        sequential, input_dim=input_dim, dc_decomp=True, n_subgrad=n_subgrad, IBP=True, forward=True
    )

    for layer in monotonic_model.layers:
        if hasattr(layer, "n_subgrad"):
            assert layer.n_subgrad == n_subgrad

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


@pytest.mark.parametrize("odd, n_subgrad", [(0, 0), (1, 0), (0, 1), (1, 1), (0, 5), (1, 5)])
def test_clone_sequential_model_multid_box_nodc(odd, n_subgrad):

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_
    input_dim = x.shape[-1]

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(
        sequential, input_dim=input_dim, n_subgrad=n_subgrad, IBP=True, forward=True
    )

    for layer in monotonic_model.layers:
        if hasattr(layer, "n_subgrad"):
            assert layer.n_subgrad == n_subgrad

    assert isinstance(monotonic_model, DecomonModel)
    assert not isinstance(sequential, DecomonModel)

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = monotonic_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")


@pytest.mark.parametrize("odd, n_subgrad", [(0, 0), (1, 0), (0, 1), (1, 1), (0, 5), (1, 5)])
def test_convert_sequential_model_multid_box(odd, n_subgrad):

    inputs = get_tensor_decomposition_multid_box(odd)
    inputs_ = get_standard_values_multid_box_convert(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs_

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = convert(sequential, dc_decomp=True, n_subgrad=n_subgrad, IBP=True, forward=True)

    for layer in monotonic_model.layers:
        if hasattr(layer, "n_subgrad"):
            assert layer.n_subgrad == n_subgrad

    assert isinstance(monotonic_model, DecomonModel)
    assert not isinstance(sequential, DecomonModel)

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = monotonic_model(inputs[1:3] + inputs[-2:])

    f_clone = K.function(inputs[1:3] + inputs[-2:], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_clone(inputs_[1:3] + inputs_[-2:])
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


@pytest.mark.parametrize("odd, n_subgrad", [(0, 0), (1, 0), (0, 1), (1, 1), (0, 5), (1, 5)])
def test_clone_sequential_model_multid_box_nodc_0(odd, n_subgrad):

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    # inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box_convert(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = convert(sequential, n_subgrad=n_subgrad, IBP=True, forward=True)

    for layer in monotonic_model.layers:
        if hasattr(layer, "n_subgrad"):
            assert layer.n_subgrad == n_subgrad

    assert isinstance(monotonic_model, DecomonModel)
    assert not isinstance(sequential, DecomonModel)

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = monotonic_model(inputs[1:3])

    f_clone = K.function(inputs[1:3], output)
    y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_clone(inputs_[1:3])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc")


# testing the different options in the forward mode
@pytest.mark.parametrize(
    "n, IBP, forward",
    [(0, True, False), (1, False, True)],
)
def test_clone_sequential_model_1d_box_forward_prop(n, IBP, forward):
    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(sequential, input_dim=1, dc_decomp=True, IBP=IBP, forward=forward)

    assert isinstance(monotonic_model, DecomonModel)
    assert not isinstance(sequential, DecomonModel)

    inputs = get_tensor_decomposition_1d_box()
    inputs_ = get_standart_values_1d_box(n)

    if not IBP and forward:
        inputs_ = [inputs_[i] for i in [0, 1, 2, 4, 5, 7, 8, 9, 10]]
        inputs = [inputs[i] for i in [0, 1, 2, 4, 5, 7, 8, 9, 10]]
    if IBP and not forward:
        inputs_ = [inputs_[i] for i in [0, 1, 2, 3, 6, 9, 10]]
        inputs = [inputs[i] for i in [0, 1, 2, 3, 6, 9, 10]]

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = monotonic_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    f_clone(inputs_[1:])


@pytest.mark.parametrize(
    "n, IBP, forward, dc_decomp, mode",
    [
        (0, True, False, True, "forward"),
        (0, False, True, True, "forward"),
        (0, True, False, False, "forward"),
        (0, False, True, False, "forward"),
        (0, True, True, False, "forward"),
        (0, True, True, True, "forward"),
        (1, True, False, True, "forward"),
        (1, False, True, True, "forward"),
        (1, True, False, False, "forward"),
        (1, False, True, False, "forward"),
        (1, True, True, False, "forward"),
        (1, True, True, True, "forward"),
        (0, True, False, False, "backward"),
        (0, False, True, False, "backward"),
        (0, True, True, False, "backward"),
    ],
)
def test_convert_model_1d_box_forward_prop(n, IBP, forward, dc_decomp, mode):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="linear", input_dim=1))
    sequential.add(Dense(3, activation="linear"))

    monotonic_model = convert(sequential, dc_decomp=dc_decomp, IBP=IBP, forward=forward, mode=mode)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=dc_decomp)

    if dc_decomp:
        h, g = inputs[-2:]
        h_i, g_i = inputs_[-2:]

    x = inputs[0]
    z = inputs[2]

    x_i = inputs_[0]
    z_i = inputs_[2]

    if dc_decomp:
        output = monotonic_model([x, z, h, g])

        f_clone = K.function([x, z, h, g], output)
        output_ = f_clone([x_i, z_i, h_i, g_i])
    else:
        output = monotonic_model([x, z])

        f_clone = K.function([x, z], output)
        output_ = f_clone([x_i, z_i])
