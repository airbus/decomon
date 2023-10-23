import keras.ops as K
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

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
    dc_decomp = True

    # inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs_)

    if finetune_odd is not None:
        finetune_odd = K.convert_to_tensor(finetune_odd, dtype="float{}".format(floatx))

    # decomon output
    output = func(inputs_for_mode, mode=mode, axis=axis, finetune_lower=finetune_odd)
    f_pooling = helpers.function(inputs, output)
    w_, b_ = f_pooling(inputs_)

    # reference output
    output_ref_ = np.max(input_ref_, axis=axis)

    assert_almost_equal(
        minmax(np.clip(np.sum(w_ * input_ref_, axis) + b_ - output_ref_, clipmin, clipmax)),
        0.0,
        decimal=decimal,
        err_msg=f"linear hull for bounding max",
    )
