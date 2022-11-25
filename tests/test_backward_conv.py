# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Input

from decomon.backward_layers.backward_layers import get_backward
from decomon.layers.decomon_layers import DecomonConv2D

from . import (
    assert_output_properties_box_linear,
    get_standard_values_images_box,
    get_tensor_decomposition_images_box,
)


@pytest.mark.parametrize(
    "data_format, odd, m_0, m_1, activation, padding, use_bias, mode, floatx, previous",
    [
        ("channels_last", 0, 0, 1, None, "same", True, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, None, "same", True, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, None, "same", False, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, None, "same", False, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "hybrid", 32, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "hybrid", 32, True),
        ("channels_last", 0, 0, 1, None, "same", True, "forward", 32, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "forward", 32, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "forward", 32, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "forward", 32, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "forward", 32, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "forward", 32, True),
        ("channels_first", 0, 0, 1, None, "same", True, "forward", 32, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "forward", 32, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "forward", 32, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "forward", 32, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "forward", 32, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "forward", 32, True),
        ("channels_last", 0, 0, 1, None, "same", False, "forward", 32, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "forward", 32, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "forward", 32, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "forward", 32, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "forward", 32, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "forward", 32, True),
        ("channels_first", 0, 0, 1, None, "same", False, "forward", 32, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "forward", 32, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "forward", 32, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "forward", 32, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "forward", 32, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "forward", 32, True),
        ("channels_last", 0, 0, 1, None, "same", True, "ibp", 32, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "ibp", 32, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "ibp", 32, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "ibp", 32, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "ibp", 32, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "ibp", 32, True),
        ("channels_first", 0, 0, 1, None, "same", True, "ibp", 32, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "ibp", 32, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "ibp", 32, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "ibp", 32, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "ibp", 32, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "ibp", 32, True),
        ("channels_last", 0, 0, 1, None, "same", False, "ibp", 32, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "ibp", 32, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "ibp", 32, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "ibp", 32, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "ibp", 32, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "ibp", 32, True),
        ("channels_first", 0, 0, 1, None, "same", False, "ibp", 32, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "ibp", 32, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "ibp", 32, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "ibp", 32, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "ibp", 32, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "ibp", 32, True),
        ("channels_last", 0, 0, 1, None, "same", True, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, None, "same", True, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, None, "same", False, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, None, "same", False, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "hybrid", 64, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "hybrid", 64, True),
        ("channels_last", 0, 0, 1, None, "same", True, "forward", 64, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "forward", 64, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "forward", 64, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "forward", 64, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "forward", 64, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "forward", 64, True),
        ("channels_first", 0, 0, 1, None, "same", True, "forward", 64, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "forward", 64, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "forward", 64, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "forward", 64, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "forward", 64, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "forward", 64, True),
        ("channels_last", 0, 0, 1, None, "same", False, "forward", 64, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "forward", 64, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "forward", 64, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "forward", 64, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "forward", 64, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "forward", 64, True),
        ("channels_first", 0, 0, 1, None, "same", False, "forward", 64, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "forward", 64, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "forward", 64, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "forward", 64, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "forward", 64, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "forward", 64, True),
        ("channels_last", 0, 0, 1, None, "same", True, "ibp", 64, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "ibp", 64, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "ibp", 64, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "ibp", 64, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "ibp", 64, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "ibp", 64, True),
        ("channels_first", 0, 0, 1, None, "same", True, "ibp", 64, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "ibp", 64, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "ibp", 64, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "ibp", 64, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "ibp", 64, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "ibp", 64, True),
        ("channels_last", 0, 0, 1, None, "same", False, "ibp", 64, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "ibp", 64, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "ibp", 64, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "ibp", 64, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "ibp", 64, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "ibp", 64, True),
        ("channels_first", 0, 0, 1, None, "same", False, "ibp", 64, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "ibp", 64, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "ibp", 64, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "ibp", 64, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "ibp", 64, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "ibp", 64, True),
        ("channels_last", 0, 0, 1, None, "same", True, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, None, "same", True, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, None, "same", False, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, None, "same", False, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "hybrid", 16, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "hybrid", 16, True),
        ("channels_last", 0, 0, 1, None, "same", True, "forward", 16, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "forward", 16, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "forward", 16, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "forward", 16, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "forward", 16, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "forward", 16, True),
        ("channels_first", 0, 0, 1, None, "same", True, "forward", 16, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "forward", 16, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "forward", 16, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "forward", 16, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "forward", 16, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "forward", 16, True),
        ("channels_last", 0, 0, 1, None, "same", False, "forward", 16, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "forward", 16, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "forward", 16, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "forward", 16, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "forward", 16, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "forward", 16, True),
        ("channels_first", 0, 0, 1, None, "same", False, "forward", 16, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "forward", 16, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "forward", 16, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "forward", 16, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "forward", 16, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "forward", 16, True),
        ("channels_last", 0, 0, 1, None, "same", True, "ibp", 16, True),
        ("channels_last", 0, 0, 1, "linear", "same", True, "ibp", 16, True),
        ("channels_last", 0, 0, 1, "relu", "same", True, "ibp", 16, True),
        ("channels_last", 0, 0, 1, None, "valid", True, "ibp", 16, True),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "ibp", 16, True),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "ibp", 16, True),
        ("channels_first", 0, 0, 1, None, "same", True, "ibp", 16, True),
        ("channels_first", 0, 0, 1, "linear", "same", True, "ibp", 16, True),
        ("channels_first", 0, 0, 1, "relu", "same", True, "ibp", 16, True),
        ("channels_first", 0, 0, 1, None, "valid", True, "ibp", 16, True),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "ibp", 16, True),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "ibp", 16, True),
        ("channels_last", 0, 0, 1, None, "same", False, "ibp", 16, True),
        ("channels_last", 0, 0, 1, "linear", "same", False, "ibp", 16, True),
        ("channels_last", 0, 0, 1, "relu", "same", False, "ibp", 16, True),
        ("channels_last", 0, 0, 1, None, "valid", False, "ibp", 16, True),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "ibp", 16, True),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "ibp", 16, True),
        ("channels_first", 0, 0, 1, None, "same", False, "ibp", 16, True),
        ("channels_first", 0, 0, 1, "linear", "same", False, "ibp", 16, True),
        ("channels_first", 0, 0, 1, "relu", "same", False, "ibp", 16, True),
        ("channels_first", 0, 0, 1, None, "valid", False, "ibp", 16, True),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "ibp", 16, True),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "ibp", 16, True),
        ("channels_last", 0, 0, 1, None, "same", True, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, None, "same", True, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, None, "same", False, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, None, "same", False, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "hybrid", 32, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "hybrid", 32, False),
        ("channels_last", 0, 0, 1, None, "same", True, "forward", 32, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "forward", 32, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "forward", 32, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "forward", 32, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "forward", 32, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "forward", 32, False),
        ("channels_first", 0, 0, 1, None, "same", True, "forward", 32, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "forward", 32, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "forward", 32, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "forward", 32, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "forward", 32, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "forward", 32, False),
        ("channels_last", 0, 0, 1, None, "same", False, "forward", 32, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "forward", 32, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "forward", 32, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "forward", 32, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "forward", 32, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "forward", 32, False),
        ("channels_first", 0, 0, 1, None, "same", False, "forward", 32, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "forward", 32, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "forward", 32, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "forward", 32, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "forward", 32, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "forward", 32, False),
        ("channels_last", 0, 0, 1, None, "same", True, "ibp", 32, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "ibp", 32, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "ibp", 32, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "ibp", 32, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "ibp", 32, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "ibp", 32, False),
        ("channels_first", 0, 0, 1, None, "same", True, "ibp", 32, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "ibp", 32, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "ibp", 32, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "ibp", 32, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "ibp", 32, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "ibp", 32, False),
        ("channels_last", 0, 0, 1, None, "same", False, "ibp", 32, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "ibp", 32, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "ibp", 32, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "ibp", 32, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "ibp", 32, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "ibp", 32, False),
        ("channels_first", 0, 0, 1, None, "same", False, "ibp", 32, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "ibp", 32, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "ibp", 32, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "ibp", 32, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "ibp", 32, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "ibp", 32, False),
        ("channels_last", 0, 0, 1, None, "same", True, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, None, "same", True, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, None, "same", False, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, None, "same", False, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "hybrid", 64, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "hybrid", 64, False),
        ("channels_last", 0, 0, 1, None, "same", True, "forward", 64, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "forward", 64, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "forward", 64, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "forward", 64, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "forward", 64, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "forward", 64, False),
        ("channels_first", 0, 0, 1, None, "same", True, "forward", 64, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "forward", 64, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "forward", 64, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "forward", 64, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "forward", 64, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "forward", 64, False),
        ("channels_last", 0, 0, 1, None, "same", False, "forward", 64, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "forward", 64, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "forward", 64, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "forward", 64, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "forward", 64, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "forward", 64, False),
        ("channels_first", 0, 0, 1, None, "same", False, "forward", 64, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "forward", 64, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "forward", 64, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "forward", 64, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "forward", 64, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "forward", 64, False),
        ("channels_last", 0, 0, 1, None, "same", True, "ibp", 64, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "ibp", 64, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "ibp", 64, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "ibp", 64, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "ibp", 64, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "ibp", 64, False),
        ("channels_first", 0, 0, 1, None, "same", True, "ibp", 64, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "ibp", 64, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "ibp", 64, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "ibp", 64, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "ibp", 64, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "ibp", 64, False),
        ("channels_last", 0, 0, 1, None, "same", False, "ibp", 64, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "ibp", 64, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "ibp", 64, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "ibp", 64, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "ibp", 64, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "ibp", 64, False),
        ("channels_first", 0, 0, 1, None, "same", False, "ibp", 64, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "ibp", 64, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "ibp", 64, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "ibp", 64, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "ibp", 64, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "ibp", 64, False),
        ("channels_last", 0, 0, 1, None, "same", True, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, None, "same", True, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, None, "same", False, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, None, "same", False, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "hybrid", 16, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "hybrid", 16, False),
        ("channels_last", 0, 0, 1, None, "same", True, "forward", 16, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "forward", 16, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "forward", 16, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "forward", 16, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "forward", 16, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "forward", 16, False),
        ("channels_first", 0, 0, 1, None, "same", True, "forward", 16, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "forward", 16, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "forward", 16, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "forward", 16, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "forward", 16, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "forward", 16, False),
        ("channels_last", 0, 0, 1, None, "same", False, "forward", 16, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "forward", 16, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "forward", 16, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "forward", 16, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "forward", 16, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "forward", 16, False),
        ("channels_first", 0, 0, 1, None, "same", False, "forward", 16, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "forward", 16, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "forward", 16, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "forward", 16, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "forward", 16, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "forward", 16, False),
        ("channels_last", 0, 0, 1, None, "same", True, "ibp", 16, False),
        ("channels_last", 0, 0, 1, "linear", "same", True, "ibp", 16, False),
        ("channels_last", 0, 0, 1, "relu", "same", True, "ibp", 16, False),
        ("channels_last", 0, 0, 1, None, "valid", True, "ibp", 16, False),
        ("channels_last", 0, 0, 1, "linear", "valid", True, "ibp", 16, False),
        ("channels_last", 0, 0, 1, "relu", "valid", True, "ibp", 16, False),
        ("channels_first", 0, 0, 1, None, "same", True, "ibp", 16, False),
        ("channels_first", 0, 0, 1, "linear", "same", True, "ibp", 16, False),
        ("channels_first", 0, 0, 1, "relu", "same", True, "ibp", 16, False),
        ("channels_first", 0, 0, 1, None, "valid", True, "ibp", 16, False),
        ("channels_first", 0, 0, 1, "linear", "valid", True, "ibp", 16, False),
        ("channels_first", 0, 0, 1, "relu", "valid", True, "ibp", 16, False),
        ("channels_last", 0, 0, 1, None, "same", False, "ibp", 16, False),
        ("channels_last", 0, 0, 1, "linear", "same", False, "ibp", 16, False),
        ("channels_last", 0, 0, 1, "relu", "same", False, "ibp", 16, False),
        ("channels_last", 0, 0, 1, None, "valid", False, "ibp", 16, False),
        ("channels_last", 0, 0, 1, "linear", "valid", False, "ibp", 16, False),
        ("channels_last", 0, 0, 1, "relu", "valid", False, "ibp", 16, False),
        ("channels_first", 0, 0, 1, None, "same", False, "ibp", 16, False),
        ("channels_first", 0, 0, 1, "linear", "same", False, "ibp", 16, False),
        ("channels_first", 0, 0, 1, "relu", "same", False, "ibp", 16, False),
        ("channels_first", 0, 0, 1, None, "valid", False, "ibp", 16, False),
        ("channels_first", 0, 0, 1, "linear", "valid", False, "ibp", 16, False),
        ("channels_first", 0, 0, 1, "relu", "valid", False, "ibp", 16, False),
    ],
)
def test_Decomon_conv_box(data_format, odd, m_0, m_1, activation, padding, use_bias, mode, floatx, previous):

    if data_format == "channels_first" and not len(K._get_available_gpus()):
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    layer = DecomonConv2D(
        10,
        kernel_size=(3, 3),
        activation=activation,
        dc_decomp=False,
        padding=padding,
        use_bias=use_bias,
        mode=mode,
        dtype=K.floatx(),
    )

    inputs = get_tensor_decomposition_images_box(data_format, odd, dc_decomp=False)
    inputs_ = get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=False)
    x, y, z_, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    if mode == "hybrid":
        input_mode = inputs[2:]
        output = layer(input_mode)
        z_0, u_c_0, _, _, l_c_0, _, _ = output
    if mode == "forward":
        input_mode = [inputs[2], inputs[4], inputs[5], inputs[7], inputs[8]]
        output = layer(input_mode)
        z_0, _, _, _, _ = output
    if mode == "ibp":
        input_mode = [inputs[3], inputs[6]]
        output = layer(input_mode)
        u_c_0, l_c_0 = output

    output_shape = np.prod(output[-1].shape[1:])

    w_out = Input((output_shape, output_shape))
    b_out = Input((output_shape,))
    # get backward layer
    layer_backward = get_backward(layer, previous=previous)

    if previous:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode + [w_out, b_out, w_out, b_out])
        f_conv = K.function(inputs + [w_out, b_out], [w_out_u, b_out_u, w_out_l, b_out_l])
        w_init = np.concatenate([np.diag([1.0] * output_shape)[None]] * len(x)).reshape(
            (-1, output_shape, output_shape)
        )
        b_init = np.zeros((len(x), output_shape))
        output_ = f_conv(inputs_ + [w_init, b_init])

    else:
        w_out_u, b_out_u, w_out_l, b_out_l = layer_backward(input_mode)
        f_conv = K.function(inputs, [w_out_u, b_out_u, w_out_l, b_out_l])
        output_ = f_conv(inputs_)

    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()

    w_u_, b_u_, w_l_, b_l_ = output_

    w_u_ = w_u_[:, None]
    w_l_ = w_l_[:, None]
    b_u_ = b_u_[:, None]
    b_l_ = b_l_[:, None]

    # step 1: flatten W_u
    W_u = W_u.reshape((len(W_u), W_u.shape[1], -1))
    W_l = W_l.reshape((len(W_l), W_l.shape[1], -1))

    w_r_u = np.sum(np.maximum(0.0, w_u_) * np.expand_dims(W_u, -1), 2) + np.sum(
        np.minimum(0.0, w_u_) * np.expand_dims(W_l, -1), 2
    )
    w_r_l = np.sum(np.maximum(0.0, w_l_) * np.expand_dims(W_l, -1), 2) + np.sum(
        np.minimum(0.0, w_l_) * np.expand_dims(W_u, -1), 2
    )

    b_l = b_l.reshape((len(b_l), -1))
    b_u = b_u.reshape((len(b_u), -1))

    b_r_u = (
        np.sum(np.maximum(0, w_u_[:, 0]) * np.expand_dims(b_u, -1), 1)[:, None]
        + np.sum(np.minimum(0, w_u_[:, 0]) * np.expand_dims(b_l, -1), 1)[:, None]
        + b_u_
    )
    b_r_l = (
        np.sum(np.maximum(0, w_l_[:, 0]) * np.expand_dims(b_l, -1), 1)[:, None]
        + np.sum(np.minimum(0, w_l_[:, 0]) * np.expand_dims(b_u, -1), 1)[:, None]
        + b_l_
    )

    assert_output_properties_box_linear(x, None, z_[:, 0], z_[:, 1], None, w_r_u, b_r_u, None, w_r_l, b_r_l, "nodc")

    K.set_floatx("float32")
    K.set_epsilon(eps)
