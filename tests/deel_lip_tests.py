from __future__ import absolute_import

import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from deel.lip.activations import GroupSort2
from numpy.testing import assert_allclose, assert_almost_equal
from tensorflow.keras.layers import Dense

from decomon.layers.activations import sigmoid, softplus, softsign, tanh

from . import (
    assert_output_properties_box,
    assert_output_properties_box_nodc,
    get_standard_values_multid_box,
    get_standart_values_1d_box,
    get_tensor_decomposition_1d_box,
    get_tensor_decomposition_multid_box,
)

"""
def test_group_sort_2_activation(n):

    # create a layer with GroupSort2 as activation
    layer = Dense(1, activation=GroupSort2())
"""
