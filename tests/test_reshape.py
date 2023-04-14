import numpy as np
import pytest
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Permute, Reshape

from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import to_decomon
from decomon.layers.decomon_reshape import DecomonPermute, DecomonReshape


def test_Decomon_reshape_box(mode, floatx, helpers):
    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = helpers.get_tensor_decomposition_images_box("channels_last", odd)
    inputs_ = helpers.get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    target_shape = (np.prod(y.shape[1:]),)
    y_ = np.reshape(inputs_[1], (-1, target_shape[0]))

    decomon_layer = DecomonReshape((target_shape), dc_decomp=True, mode=mode, dtype=K.floatx())

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = decomon_layer(inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = decomon_layer([z, W_u, b_u, W_l, b_l, h, g])
    elif mode == ForwardMode.IBP:
        output = decomon_layer([u_c, l_c, h, g])
    else:
        raise ValueError("Unknown mode.")

    f_reshape = K.function(inputs[2:], output)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_reshape(inputs_[2:])
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_reshape(inputs_[2:])
        u_c_ = None
        l_c_ = None
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_reshape(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
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


def test_Decomon_reshape_box_nodc(mode, floatx, helpers):
    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = helpers.get_tensor_decomposition_images_box("channels_last", odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    target_shape = (np.prod(y.shape[1:]),)
    y_ = np.reshape(inputs_[1], (-1, target_shape[0]))

    decomon_layer = DecomonReshape((target_shape), dc_decomp=False, mode=mode, dtype=K.floatx())

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = decomon_layer(inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = decomon_layer([z, W_u, b_u, W_l, b_l])
    elif mode == ForwardMode.IBP:
        output = decomon_layer([u_c, l_c])
    else:
        raise ValueError("Unknown mode.")

    f_reshape = K.function(inputs[2:], output)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_reshape(inputs_[2:])
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = f_reshape(inputs_[2:])
        u_c_ = None
        l_c_ = None
    elif mode == ForwardMode.IBP:
        u_c_, l_c_ = f_reshape(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
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


def test_Decomon_reshape_to_decomon_box(shared, floatx, helpers):
    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 4
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = helpers.get_tensor_decomposition_images_box("channels_last", odd)
    inputs_ = helpers.get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    target_shape = (np.prod(y.shape[1:]),)

    reshape_ref = Reshape(target_shape, dtype=K.floatx())
    output_ref = reshape_ref(inputs[1])

    input_dim = x_.shape[-1]
    decomon_layer = to_decomon(reshape_ref, input_dim, dc_decomp=True, shared=shared)

    output = decomon_layer(inputs[2:])

    f_ref = K.function(inputs, output_ref)

    f_reshape = K.function(inputs[2:], output)
    y_ref = f_ref(inputs_)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_reshape(inputs_[2:])

    helpers.assert_output_properties_box(
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
def test_Decomon_permute_box(mode, floatx, helpers):
    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = helpers.get_tensor_decomposition_images_box("channels_last", odd)
    inputs_ = helpers.get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    n_dim = len(y.shape) - 1
    target_shape = np.random.permutation(n_dim) + 1
    target_shape_ = tuple([0] + list(target_shape))

    y_ = np.transpose(inputs_[1], target_shape_)

    decomon_layer = DecomonPermute(target_shape, dc_decomp=True, mode=mode, dtype=K.floatx())

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = decomon_layer(inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = decomon_layer([z, W_u, b_u, W_l, b_l, h, g])
    elif mode == ForwardMode.IBP:
        output = decomon_layer([u_c, l_c, h, g])
    else:
        raise ValueError("Unknown mode.")

    f_permute = K.function(inputs[2:], output)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_permute(inputs_[2:])
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_permute(inputs_[2:])
        u_c_ = None
        l_c_ = None
    elif mode == ForwardMode.IBP:
        u_c_, l_c_, h_, g_ = f_permute(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
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


def test_Decomon_permute_box_nodc(mode, floatx, helpers):
    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = helpers.get_tensor_decomposition_images_box("channels_last", odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    n_dim = len(y.shape) - 1
    target_shape = np.random.permutation(n_dim) + 1
    target_shape_ = tuple([0] + list(target_shape))

    y_ = np.transpose(inputs_[1], target_shape_)

    decomon_layer = DecomonPermute(target_shape, dc_decomp=False, mode=mode, dtype=K.floatx())

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = decomon_layer(inputs[2:])
    elif mode == ForwardMode.AFFINE:
        output = decomon_layer([z, W_u, b_u, W_l, b_l])
    elif mode == ForwardMode.IBP:
        output = decomon_layer([u_c, l_c])
    else:
        raise ValueError("Unknown mode.")

    f_permute = K.function(inputs[2:], output)
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_permute(inputs_[2:])
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = f_permute(inputs_[2:])
        u_c_ = None
        l_c_ = None
    elif mode == ForwardMode.IBP:
        u_c_, l_c_ = f_permute(inputs_[2:])
        w_u_, b_u_, w_l_, b_l_ = [None] * 4
    else:
        raise ValueError("Unknown mode.")

    helpers.assert_output_properties_box(
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


def test_Decomon_permute_to_decomon_box(shared, floatx, helpers):
    odd, m_0, m_1 = 0, 0, 1

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 4
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 1

    inputs = helpers.get_tensor_decomposition_images_box("channels_last", odd)
    inputs_ = helpers.get_standard_values_images_box("channels_last", odd, m0=m_0, m1=m_1)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    n_dim = len(y.shape) - 1
    target_shape = np.random.permutation(n_dim) + 1

    permute_ref = Permute(target_shape)
    output_ref = permute_ref(inputs[1])

    input_dim = x_.shape[-1]
    decomon_layer = to_decomon(permute_ref, input_dim, dc_decomp=True, shared=shared)
    output = decomon_layer(inputs[2:])

    f_ref = K.function(inputs, output_ref)

    f_permute = K.function(inputs[2:], output)
    y_ref = f_ref(inputs_)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_permute(inputs_[2:])

    helpers.assert_output_properties_box(
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
