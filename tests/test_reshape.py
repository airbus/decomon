import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Permute, Reshape

from decomon.layers.decomon_layers import to_monotonic
from decomon.layers.decomon_reshape import DecomonPermute, DecomonReshape

from . import (
    assert_output_properties_box,
    assert_output_properties_box_linear,
    get_standard_values_images_box,
    get_tensor_decomposition_images_box,
)


@pytest.mark.parametrize(
    "odd, m_0, m_1, mode, floatx",
    [
        (0, 0, 1, "hybrid", 16),
        (0, 0, 1, "forward", 32),
        (0, 0, 1, "ibp", 32),
        (0, 0, 1, "hybrid", 64),
        (0, 0, 1, "forward", 64),
        (0, 0, 1, "ibp", 64),
        (0, 0, 1, "hybrid", 16),
        (0, 0, 1, "forward", 16),
        (0, 0, 1, "ibp", 16),
    ],
)
def test_Decomon_reshape_box(odd, m_0, m_1, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    # monotonic_layer = DecomonConv2D(10, kernel_size=(3, 3), activation="relu", dc_decomp=True, mode=mode,
    #                                data_format=data_format)

    inputs = get_tensor_decomposition_images_box("channels_last", odd)
    inputs_ = get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    target_shape = (np.prod(y.shape[1:]),)
    y_ = np.reshape(inputs_[1], (-1, target_shape[0]))

    monotonic_layer = DecomonReshape((target_shape), dc_decomp=True, mode=mode, dtype=K.floatx())

    if mode == "hybrid":
        output = monotonic_layer(inputs[2:])
    if mode == "forward":
        output = monotonic_layer([z, W_u, b_u, W_l, b_l, h, g])
    if mode == "ibp":
        output = monotonic_layer([u_c, l_c, h, g])

    f_reshape = K.function(inputs[2:], output)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_reshape(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_reshape(inputs_[2:])
        u_c_ = None
        l_c_ = None
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_reshape(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4
    assert_output_properties_box(
        x_,
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
        "reshape_{}_{}_{}".format(odd, m_0, m_1),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, m_0, m_1, mode, floatx",
    [
        (0, 0, 1, "hybrid", 32),
        (0, 0, 1, "forward", 32),
        (0, 0, 1, "ibp", 32),
        (0, 0, 1, "hybrid", 64),
        (0, 0, 1, "forward", 64),
        (0, 0, 1, "ibp", 64),
        (0, 0, 1, "hybrid", 16),
        (0, 0, 1, "forward", 16),
        (0, 0, 1, "ibp", 16),
    ],
)
def test_Decomon_reshape_box_nodc(odd, m_0, m_1, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    # monotonic_layer = DecomonConv2D(10, kernel_size=(3, 3), activation="relu", dc_decomp=True, mode=mode,
    #                                data_format=data_format)

    inputs = get_tensor_decomposition_images_box("channels_last", odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    target_shape = (np.prod(y.shape[1:]),)
    y_ = np.reshape(inputs_[1], (-1, target_shape[0]))

    monotonic_layer = DecomonReshape((target_shape), dc_decomp=False, mode=mode, dtype=K.floatx())

    if mode == "hybrid":
        output = monotonic_layer(inputs[2:])
    if mode == "forward":
        output = monotonic_layer([z, W_u, b_u, W_l, b_l])
    if mode == "ibp":
        output = monotonic_layer([u_c, l_c])

    f_reshape = K.function(inputs[2:], output)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_reshape(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_reshape(inputs_[2:])
        u_c_ = None
        l_c_ = None
    if mode == "ibp":
        u_c_, l_c_ = f_reshape(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_,
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
        "reshape_{}_{}_{}".format(odd, m_0, m_1),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, m_0, m_1, shared, floatx",
    [
        (0, 0, 1, False, 32),
        (0, 0, 1, True, 32),
        (0, 0, 1, False, 64),
        (0, 0, 1, True, 64),
        (0, 0, 1, False, 16),
        (0, 0, 1, True, 16),
    ],
)
def test_Decomon_reshape_to_monotonic_box(odd, m_0, m_1, shared, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 4
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = get_tensor_decomposition_images_box("channels_last", odd)
    inputs_ = get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    target_shape = (np.prod(y.shape[1:]),)

    reshape_ref = Reshape(target_shape, dtype=K.floatx())
    output_ref = reshape_ref(inputs[1])

    input_dim = x_.shape[-1]
    monotonic_layer = to_monotonic(reshape_ref, input_dim, dc_decomp=True, shared=shared)

    output = monotonic_layer[0](inputs[2:])
    if len(monotonic_layer) > 1:
        output = monotonic_layer[1](output)

    f_ref = K.function(inputs, output_ref)

    f_reshape = K.function(inputs[2:], output)
    y_ref = f_ref(inputs_)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_reshape(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
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
        "reshape_{}_{}_{}".format(odd, m_0, m_1),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


# permute


@pytest.mark.parametrize(
    "odd, m_0, m_1, mode, floatx",
    [
        (0, 0, 1, "hybrid", 32),
        (0, 0, 1, "forward", 32),
        (0, 0, 1, "ibp", 32),
        (0, 0, 1, "hybrid", 64),
        (0, 0, 1, "forward", 64),
        (0, 0, 1, "ibp", 64),
        (0, 0, 1, "hybrid", 16),
        (0, 0, 1, "forward", 16),
        (0, 0, 1, "ibp", 16),
    ],
)
def test_Decomon_permute_box(odd, m_0, m_1, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = get_tensor_decomposition_images_box("channels_last", odd)
    inputs_ = get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    n_dim = len(y.shape) - 1
    target_shape = np.random.permutation(n_dim) + 1
    target_shape_ = tuple([0] + list(target_shape))

    y_ = np.transpose(inputs_[1], target_shape_)

    monotonic_layer = DecomonPermute(target_shape, dc_decomp=True, mode=mode, dtype=K.floatx())

    if mode == "hybrid":
        output = monotonic_layer(inputs[2:])
    if mode == "forward":
        output = monotonic_layer([z, W_u, b_u, W_l, b_l, h, g])
    if mode == "ibp":
        output = monotonic_layer([u_c, l_c, h, g])

    f_permute = K.function(inputs[2:], output)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_permute(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_permute(inputs_[2:])
        u_c_ = None
        l_c_ = None
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_permute(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_,
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
        "reshape_{}_{}_{}".format(odd, m_0, m_1),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, m_0, m_1, mode, floatx",
    [
        (0, 0, 1, "hybrid", 32),
        (0, 0, 1, "forward", 32),
        (0, 0, 1, "ibp", 32),
        (0, 0, 1, "hybrid", 64),
        (0, 0, 1, "forward", 64),
        (0, 0, 1, "ibp", 64),
        (0, 0, 1, "hybrid", 16),
        (0, 0, 1, "forward", 16),
        (0, 0, 1, "ibp", 16),
    ],
)
def test_Decomon_permute_box_nodc(odd, m_0, m_1, mode, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = get_tensor_decomposition_images_box("channels_last", odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    n_dim = len(y.shape) - 1
    target_shape = np.random.permutation(n_dim) + 1
    target_shape_ = tuple([0] + list(target_shape))

    y_ = np.transpose(inputs_[1], target_shape_)

    monotonic_layer = DecomonPermute(target_shape, dc_decomp=False, mode=mode, dtype=K.floatx())

    if mode == "hybrid":
        output = monotonic_layer(inputs[2:])
    if mode == "forward":
        output = monotonic_layer([z, W_u, b_u, W_l, b_l])
    if mode == "ibp":
        output = monotonic_layer([u_c, l_c])

    f_permute = K.function(inputs[2:], output)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_permute(inputs_[2:])
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_permute(inputs_[2:])
        u_c_ = None
        l_c_ = None
    if mode == "ibp":
        u_c_, l_c_ = f_permute(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4

    assert_output_properties_box(
        x_,
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
        "reshape_{}_{}_{}".format(odd, m_0, m_1),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


@pytest.mark.parametrize(
    "odd, m_0, m_1, shared, floatx",
    [
        (0, 0, 1, False, 32),
        (0, 0, 1, True, 32),
        (0, 0, 1, False, 64),
        (0, 0, 1, True, 64),
        (0, 0, 1, False, 16),
        (0, 0, 1, True, 16),
    ],
)
def test_Decomon_permute_to_monotonic_box(odd, m_0, m_1, shared, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 4
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = get_tensor_decomposition_images_box("channels_last", odd)
    inputs_ = get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    n_dim = len(y.shape) - 1
    target_shape = np.random.permutation(n_dim) + 1

    permute_ref = Permute(target_shape)
    output_ref = permute_ref(inputs[1])

    input_dim = x_.shape[-1]
    monotonic_layer = to_monotonic(permute_ref, input_dim, dc_decomp=True, shared=shared)

    output = monotonic_layer[0](inputs[2:])
    if len(monotonic_layer) > 1:
        output = monotonic_layer[1](output)

    f_ref = K.function(inputs, output_ref)

    f_permute = K.function(inputs[2:], output)
    y_ref = f_ref(inputs_)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_permute(inputs_[2:])

    assert_output_properties_box(
        x_,
        y_ref,
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
        "reshape_{}_{}_{}".format(odd, m_0, m_1),
        decimal=decimal,
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
