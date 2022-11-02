from __future__ import absolute_import
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from . import (
    get_tensor_decomposition_1d_box,
    get_standart_values_1d_box,
    assert_output_properties_box,
    assert_output_properties_box_nodc,
    get_tensor_decomposition_multid_box,
    get_standard_values_multid_box,
)
import tensorflow.python.keras.backend as K

from decomon.layers.activations import sigmoid, tanh, softsign, softplus

from deel.lip.activations import GroupSort2
from tensorflow.keras.layers import Dense

"""
def test_group_sort_2_activation(n):

    # create a layer with GroupSort2 as activation
    layer = Dense(1, activation=GroupSort2())
"""
