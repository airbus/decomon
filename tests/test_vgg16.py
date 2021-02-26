from __future__ import absolute_import
import pytest
import numpy as np
import tensorflow.python.keras.backend as K
from decomon.layers.decomon_layers import DecomonDense, to_monotonic
from tensorflow.keras.layers import Dense
from . import get_tensor_decomposition_1d, get_standart_values_1d, assert_output_properties, get_standard_values_multid, get_tensor_decomposition_multid
import tensorflow.python.keras.backend as K

from decomon.applications.vgg16 import convert_to_monotonic_VGG16


def test_vgg_16(normalize=True):

    model = convert_to_monotonic_VGG16(normalized_input=normalize)