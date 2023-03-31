import numpy as np
import pytest
import tensorflow as tf
import tensorflow.python.keras.backend as K
from numpy.testing import assert_almost_equal

from decomon.layers.core import ForwardMode
from decomon.layers.utils_pooling import (
    get_lower_linear_hull_max,
    get_upper_linear_hull_max,
)


def test_get_upper_linear_hull_max(mode, floatx, axis, helpers):

    odd, m_0, m_1 = 0, 0, 1
    data_format = "channels_last"

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z = inputs_[:3]

    layer = get_upper_linear_hull_max

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = layer(inputs[2:], mode=mode, axis=axis)
    elif mode == ForwardMode.AFFINE:
        output = layer(
            [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8], inputs[9], inputs[10]], mode=mode, axis=axis
        )
    elif mode == ForwardMode.IBP:
        output = layer([inputs[3], inputs[6], inputs[9], inputs[10]], mode=mode, axis=axis)
    else:
        raise ValueError("Unknown mode.")

    f_pooling = K.function(inputs, output)
    w_u_, b_u_ = f_pooling(inputs_)
    y_ = np.max(y, axis=axis)

    assert_almost_equal(
        np.min(np.clip(np.sum(w_u_ * y, axis) + b_u_ - y_, 0, np.inf)),
        0.0,
        decimal=decimal,
        err_msg="linear hull for upper bounding max",
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


def test_get_lower_linear_hull_max(mode, floatx, axis, helpers):

    odd, m_0, m_1 = 0, 0, 1
    data_format = "channels_last"

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z = inputs_[:3]

    layer = get_lower_linear_hull_max

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = layer(inputs[2:], mode=mode, axis=axis)
    elif mode == ForwardMode.AFFINE:
        output = layer(
            [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8], inputs[9], inputs[10]], mode=mode, axis=axis
        )
    elif mode == ForwardMode.IBP:
        output = layer([inputs[3], inputs[6], inputs[9], inputs[10]], mode=mode, axis=axis)
    else:
        raise ValueError("Unknown mode.")

    f_pooling = K.function(inputs, output)
    w_l_, b_l_ = f_pooling(inputs_)
    y_ = np.max(y, axis=axis)

    assert_almost_equal(
        np.max(np.clip(np.sum(w_l_ * y, axis) + b_l_ - y_, -np.inf, 0.0)),
        0.0,
        decimal=decimal,
        err_msg="linear hull for lower bounding max",
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)


def test_get_lower_linear_hull_max_finetune(mode, floatx, axis, finetune_odd, helpers):

    odd, m_0, m_1 = 0, 0, 1
    data_format = "channels_last"

    finetune_odd = tf.constant(finetune_odd, dtype="float{}".format(floatx))

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z = inputs_[:3]

    layer = get_lower_linear_hull_max

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = layer(inputs[2:], mode=mode, axis=axis, finetune_lower=finetune_odd)
    elif mode == ForwardMode.AFFINE:
        output = layer(
            [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8], inputs[9], inputs[10]],
            mode=mode,
            axis=axis,
            finetune_lower=finetune_odd,
        )
    elif mode == ForwardMode.IBP:
        output = layer([inputs[3], inputs[6], inputs[9], inputs[10]], mode=mode, axis=axis, finetune_lower=finetune_odd)
    else:
        raise ValueError("Unknown mode.")

    f_pooling = K.function(inputs, output)
    w_l_, b_l_ = f_pooling(inputs_)
    y_ = np.max(y, axis=axis)

    assert_almost_equal(
        np.max(np.clip(np.sum(w_l_ * y, axis) + b_l_ - y_, -np.inf, 0.0)),
        0.0,
        decimal=decimal,
        err_msg="linear hull for lower bounding max",
    )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
