import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from numpy.testing import assert_almost_equal

from decomon.layers.core import ForwardMode
from decomon.layers.utils_pooling import (
    get_lower_linear_hull_max,
    get_upper_linear_hull_max,
)


@pytest.mark.parametrize(
    "func, minmax, clipmin, clipmax",
    [
        (get_lower_linear_hull_max, np.max, -np.inf, 0.0),
        (get_upper_linear_hull_max, np.min, 0.0, np.inf),
    ],
)
def test_get_lower_upper_linear_hull_max(
    func, minmax, clipmin, clipmax, mode, floatx, decimal, axis, finetune_odd, helpers
):

    if finetune_odd is not None and func is not get_lower_linear_hull_max:
        # skip test with finetune if not get_lower
        pytest.skip("finetune_odd is only intended for get_lower_linear_hull_max()")

    odd, m_0, m_1 = 0, 0, 1
    data_format = "channels_last"

    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1)
    x, y, z = inputs_[:3]

    if finetune_odd is not None:
        finetune_odd = tf.constant(finetune_odd, dtype="float{}".format(floatx))

    mode = ForwardMode(mode)
    if mode == ForwardMode.HYBRID:
        output = func(inputs[2:], mode=mode, axis=axis, finetune_lower=finetune_odd)
    elif mode == ForwardMode.AFFINE:
        output = func(
            [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8], inputs[9], inputs[10]],
            mode=mode,
            axis=axis,
            finetune_lower=finetune_odd,
        )
    elif mode == ForwardMode.IBP:
        output = func([inputs[3], inputs[6], inputs[9], inputs[10]], mode=mode, axis=axis, finetune_lower=finetune_odd)
    else:
        raise ValueError("Unknown mode.")

    f_pooling = K.function(inputs, output)
    w_, b_ = f_pooling(inputs_)
    y_ = np.max(y, axis=axis)

    assert_almost_equal(
        minmax(np.clip(np.sum(w_ * y, axis) + b_ - y_, clipmin, clipmax)),
        0.0,
        decimal=decimal,
        err_msg=f"linear hull for bounding max",
    )
