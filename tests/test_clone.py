# creating toy network and assess that the decomposition is correct


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Add, Average, Dense, Input, Maximum
from tensorflow.keras.models import Model, Sequential

from decomon.backward_layers.backward_layers import get_backward
from decomon.layers.decomon_layers import DecomonAdd, DecomonDense, to_monotonic
from decomon.models import clone

from . import (
    assert_output_properties_box,
    assert_output_properties_box_linear,
    get_standard_values_multid_box,
    get_standart_values_1d_box,
    get_tensor_decomposition_1d_box,
    get_tensor_decomposition_multid_box,
)
from .test_clone_forward import (
    dense_NN_1D,
    toy_network_tutorial,
    toy_struct_v0_1D,
    toy_struct_v1_1D,
    toy_struct_v2_1D,
)

"""
@pytest.mark.parametrize("n, archi, activation, sequential, use_bias, mode, method, use_input, floatx", [
    (0, [4, 3, 1], None, True, True, "ibp", "crown", True, 32),
    (1, [4, 3, 1], None, True, True, "ibp", "crown", True, 32),
    (3, [4, 3, 1], None, True, True, "ibp", "crown", True, 32),
    (5, [4, 3, 1], None, True, True, "ibp", "crown", True, 32),
    (0, [4, 3, 1], None, False, True, "ibp", "crown", True, 32),
    (1, [4, 3, 1], None, False, True, "ibp", "crown", True, 32),
    (3, [4, 3, 1], None, False, True, "ibp", "crown", True, 32),
    (5, [4, 3, 1], None, False, True, "ibp", "crown", True, 32),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown", True, 32),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown", True, 32),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown", True, 32),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown", True, 32),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown", True, 32),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown", True, 32),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown", True, 32),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown", True, 32),

    (0, [4, 3, 1], None, True, True, "forward", "crown", True, 32),
    (1, [4, 3, 1], None, True, True, "forward", "crown", True, 32),
    (3, [4, 3, 1], None, True, True, "forward", "crown", True, 32),
    (5, [4, 3, 1], None, True, True, "forward", "crown", True, 32),
    (0, [4, 3, 1], None, False, True, "forward", "crown", True, 32),
    (1, [4, 3, 1], None, False, True, "forward", "crown", True, 32),
    (3, [4, 3, 1], None, False, True, "forward", "crown", True, 32),
    (5, [4, 3, 1], None, False, True, "forward", "crown", True, 32),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "ibp", True, 32),
    (1, [4, 3, 1], None, True, True, "ibp", "ibp", True, 32),
    (3, [4, 3, 1], None, True, True, "ibp", "ibp", True, 32),
    (5, [4, 3, 1], None, True, True, "ibp", "ibp", True, 32),
    (0, [4, 3, 1], None, False, True, "ibp", "ibp", True, 32),
    (1, [4, 3, 1], None, False, True, "ibp", "ibp", True, 32),
    (3, [4, 3, 1], None, False, True, "ibp", "ibp", True, 32),
    (5, [4, 3, 1], None, False, True, "ibp", "ibp", True, 32),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "forward", True, 32),
    (1, [4, 3, 1], None, True, True, "ibp", "forward", True, 32),
    (3, [4, 3, 1], None, True, True, "ibp", "forward", True, 32),
    (5, [4, 3, 1], None, True, True, "ibp", "forward", True, 32),
    (0, [4, 3, 1], None, False, True, "ibp", "forward", True, 32),
    (1, [4, 3, 1], None, False, True, "ibp", "forward", True, 32),
    (3, [4, 3, 1], None, False, True, "ibp", "forward", True, 32),
    (5, [4, 3, 1], None, False, True, "ibp", "forward", True, 32),

    (0, [4, 3, 1], None, True, True, "hybrid", "forward", True, 32),
    (1, [4, 3, 1], None, True, True, "hybrid", "forward", True, 32),
    (3, [4, 3, 1], None, True, True, "hybrid", "forward", True, 32),
    (5, [4, 3, 1], None, True, True, "hybrid", "forward", True, 32),
    (0, [4, 3, 1], None, False, True, "hybrid", "forward", True, 32),
    (1, [4, 3, 1], None, False, True, "hybrid", "forward", True, 32),
    (3, [4, 3, 1], None, False, True, "hybrid", "forward", True, 32),
    (5, [4, 3, 1], None, False, True, "hybrid", "forward", True, 32),

    (0, [4, 3, 1], None, True, True, "forward", "forward", True, 32),
    (1, [4, 3, 1], None, True, True, "forward", "forward", True, 32),
    (3, [4, 3, 1], None, True, True, "forward", "forward", True, 32),
    (5, [4, 3, 1], None, True, True, "forward", "forward", True, 32),
    (0, [4, 3, 1], None, False, True, "forward", "forward", True, 32),
    (1, [4, 3, 1], None, False, True, "forward", "forward", True, 32),
    (3, [4, 3, 1], None, False, True, "forward", "forward", True, 32),
    (5, [4, 3, 1], None, False, True, "forward", "forward", True, 32),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 32),
    (1, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 32),
    (3, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 32),
    (5, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 32),
    (0, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 32),
    (1, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 32),
    (3, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 32),
    (5, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 32),

    (0, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 32),
    (1, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 32),
    (3, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 32),
    (5, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 32),
    (0, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 32),
    (1, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 32),
    (3, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 32),
    (5, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 32),

    (0, [4, 3, 1], None, True, True, "forward", "hybrid", True, 32),
    (1, [4, 3, 1], None, True, True, "forward", "hybrid", True, 32),
    (3, [4, 3, 1], None, True, True, "forward", "hybrid", True, 32),
    (5, [4, 3, 1], None, True, True, "forward", "hybrid", True, 32),
    (0, [4, 3, 1], None, False, True, "forward", "hybrid", True, 32),
    (1, [4, 3, 1], None, False, True, "forward", "hybrid", True, 32),
    (3, [4, 3, 1], None, False, True, "forward", "hybrid", True, 32),
    (5, [4, 3, 1], None, False, True, "forward", "hybrid", True, 32),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 32),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 32),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 32),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 32),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 32),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 32),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 32),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 32),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 32),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 32),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 32),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 32),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 32),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 32),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 32),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 32),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 32),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 32),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 32),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 32),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 32),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 32),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 32),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 32),

    (0, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 32),
    (1, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 32),
    (3, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 32),
    (5, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 32),
    (0, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 32),
    (1, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 32),
    (3, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 32),
    (5, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 32),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 32),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 32),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 32),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 32),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 32),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 32),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 32),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 32),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 32),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 32),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 32),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 32),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 32),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 32),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 32),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 32),

    (0, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 32),
    (1, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 32),
    (3, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 32),
    (5, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 32),
    (0, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 32),
    (1, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 32),
    (3, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 32),
    (5, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 32),

    ### float 16
    (0, [4, 3, 1], None, True, True, "ibp", "crown", True, 16),
    (1, [4, 3, 1], None, True, True, "ibp", "crown", True, 16),
    (3, [4, 3, 1], None, True, True, "ibp", "crown", True, 16),
    (5, [4, 3, 1], None, True, True, "ibp", "crown", True, 16),
    (0, [4, 3, 1], None, False, True, "ibp", "crown", True, 16),
    (1, [4, 3, 1], None, False, True, "ibp", "crown", True, 16),
    (3, [4, 3, 1], None, False, True, "ibp", "crown", True, 16),
    (5, [4, 3, 1], None, False, True, "ibp", "crown", True, 16),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown", True, 16),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown", True, 16),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown", True, 16),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown", True, 16),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown", True, 16),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown", True, 16),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown", True, 16),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown", True, 16),

    (0, [4, 3, 1], None, True, True, "forward", "crown", True, 16),
    (1, [4, 3, 1], None, True, True, "forward", "crown", True, 16),
    (3, [4, 3, 1], None, True, True, "forward", "crown", True, 16),
    (5, [4, 3, 1], None, True, True, "forward", "crown", True, 16),
    (0, [4, 3, 1], None, False, True, "forward", "crown", True, 16),
    (1, [4, 3, 1], None, False, True, "forward", "crown", True, 16),
    (3, [4, 3, 1], None, False, True, "forward", "crown", True, 16),
    (5, [4, 3, 1], None, False, True, "forward", "crown", True, 16),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "ibp", True, 16),
    (1, [4, 3, 1], None, True, True, "ibp", "ibp", True, 16),
    (3, [4, 3, 1], None, True, True, "ibp", "ibp", True, 16),
    (5, [4, 3, 1], None, True, True, "ibp", "ibp", True, 16),
    (0, [4, 3, 1], None, False, True, "ibp", "ibp", True, 16),
    (1, [4, 3, 1], None, False, True, "ibp", "ibp", True, 16),
    (3, [4, 3, 1], None, False, True, "ibp", "ibp", True, 16),
    (5, [4, 3, 1], None, False, True, "ibp", "ibp", True, 16),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "forward", True, 16),
    (1, [4, 3, 1], None, True, True, "ibp", "forward", True, 16),
    (3, [4, 3, 1], None, True, True, "ibp", "forward", True, 16),
    (5, [4, 3, 1], None, True, True, "ibp", "forward", True, 16),
    (0, [4, 3, 1], None, False, True, "ibp", "forward", True, 16),
    (1, [4, 3, 1], None, False, True, "ibp", "forward", True, 16),
    (3, [4, 3, 1], None, False, True, "ibp", "forward", True, 16),
    (5, [4, 3, 1], None, False, True, "ibp", "forward", True, 16),

    (0, [4, 3, 1], None, True, True, "hybrid", "forward", True, 16),
    (1, [4, 3, 1], None, True, True, "hybrid", "forward", True, 16),
    (3, [4, 3, 1], None, True, True, "hybrid", "forward", True, 16),
    (5, [4, 3, 1], None, True, True, "hybrid", "forward", True, 16),
    (0, [4, 3, 1], None, False, True, "hybrid", "forward", True, 16),
    (1, [4, 3, 1], None, False, True, "hybrid", "forward", True, 16),
    (3, [4, 3, 1], None, False, True, "hybrid", "forward", True, 16),
    (5, [4, 3, 1], None, False, True, "hybrid", "forward", True, 16),

    (0, [4, 3, 1], None, True, True, "forward", "forward", True, 16),
    (1, [4, 3, 1], None, True, True, "forward", "forward", True, 16),
    (3, [4, 3, 1], None, True, True, "forward", "forward", True, 16),
    (5, [4, 3, 1], None, True, True, "forward", "forward", True, 16),
    (0, [4, 3, 1], None, False, True, "forward", "forward", True, 16),
    (1, [4, 3, 1], None, False, True, "forward", "forward", True, 16),
    (3, [4, 3, 1], None, False, True, "forward", "forward", True, 16),
    (5, [4, 3, 1], None, False, True, "forward", "forward", True, 16),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 16),
    (1, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 16),
    (3, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 16),
    (5, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 16),
    (0, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 16),
    (1, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 16),
    (3, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 16),
    (5, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 16),

    (0, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 16),
    (1, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 16),
    (3, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 16),
    (5, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 16),
    (0, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 16),
    (1, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 16),
    (3, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 16),
    (5, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 16),

    (0, [4, 3, 1], None, True, True, "forward", "hybrid", True, 16),
    (1, [4, 3, 1], None, True, True, "forward", "hybrid", True, 16),
    (3, [4, 3, 1], None, True, True, "forward", "hybrid", True, 16),
    (5, [4, 3, 1], None, True, True, "forward", "hybrid", True, 16),
    (0, [4, 3, 1], None, False, True, "forward", "hybrid", True, 16),
    (1, [4, 3, 1], None, False, True, "forward", "hybrid", True, 16),
    (3, [4, 3, 1], None, False, True, "forward", "hybrid", True, 16),
    (5, [4, 3, 1], None, False, True, "forward", "hybrid", True, 16),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 16),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 16),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 16),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 16),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 16),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 16),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 16),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 16),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 16),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 16),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 16),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 16),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 16),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 16),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 16),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 16),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 16),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 16),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 16),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 16),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 16),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 16),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 16),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 16),

    (0, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 16),
    (1, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 16),
    (3, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 16),
    (5, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 16),
    (0, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 16),
    (1, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 16),
    (3, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 16),
    (5, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 16),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 16),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 16),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 16),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 16),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 16),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 16),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 16),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 16),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 16),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 16),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 16),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 16),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 16),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 16),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 16),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 16),

    (0, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 16),
    (1, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 16),
    (3, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 16),
    (5, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 16),
    (0, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 16),
    (1, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 16),
    (3, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 16),
    (5, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 16),


    ### float 64
    (0, [4, 3, 1], None, True, True, "ibp", "crown", True, 64),
    (1, [4, 3, 1], None, True, True, "ibp", "crown", True, 64),
    (3, [4, 3, 1], None, True, True, "ibp", "crown", True, 64),
    (5, [4, 3, 1], None, True, True, "ibp", "crown", True, 64),
    (0, [4, 3, 1], None, False, True, "ibp", "crown", True, 64),
    (1, [4, 3, 1], None, False, True, "ibp", "crown", True, 64),
    (3, [4, 3, 1], None, False, True, "ibp", "crown", True, 64),
    (5, [4, 3, 1], None, False, True, "ibp", "crown", True, 64),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown", True, 64),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown", True, 64),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown", True, 64),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown", True, 64),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown", True, 64),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown", True, 64),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown", True, 64),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown", True, 64),

    (0, [4, 3, 1], None, True, True, "forward", "crown", True, 64),
    (1, [4, 3, 1], None, True, True, "forward", "crown", True, 64),
    (3, [4, 3, 1], None, True, True, "forward", "crown", True, 64),
    (5, [4, 3, 1], None, True, True, "forward", "crown", True, 64),
    (0, [4, 3, 1], None, False, True, "forward", "crown", True, 64),
    (1, [4, 3, 1], None, False, True, "forward", "crown", True, 64),
    (3, [4, 3, 1], None, False, True, "forward", "crown", True, 64),
    (5, [4, 3, 1], None, False, True, "forward", "crown", True, 64),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "ibp", True, 64),
    (1, [4, 3, 1], None, True, True, "ibp", "ibp", True, 64),
    (3, [4, 3, 1], None, True, True, "ibp", "ibp", True, 64),
    (5, [4, 3, 1], None, True, True, "ibp", "ibp", True, 64),
    (0, [4, 3, 1], None, False, True, "ibp", "ibp", True, 64),
    (1, [4, 3, 1], None, False, True, "ibp", "ibp", True, 64),
    (3, [4, 3, 1], None, False, True, "ibp", "ibp", True, 64),
    (5, [4, 3, 1], None, False, True, "ibp", "ibp", True, 64),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "forward", True, 64),
    (1, [4, 3, 1], None, True, True, "ibp", "forward", True, 64),
    (3, [4, 3, 1], None, True, True, "ibp", "forward", True, 64),
    (5, [4, 3, 1], None, True, True, "ibp", "forward", True, 64),
    (0, [4, 3, 1], None, False, True, "ibp", "forward", True, 64),
    (1, [4, 3, 1], None, False, True, "ibp", "forward", True, 64),
    (3, [4, 3, 1], None, False, True, "ibp", "forward", True, 64),
    (5, [4, 3, 1], None, False, True, "ibp", "forward", True, 64),

    (0, [4, 3, 1], None, True, True, "hybrid", "forward", True, 64),
    (1, [4, 3, 1], None, True, True, "hybrid", "forward", True, 64),
    (3, [4, 3, 1], None, True, True, "hybrid", "forward", True, 64),
    (5, [4, 3, 1], None, True, True, "hybrid", "forward", True, 64),
    (0, [4, 3, 1], None, False, True, "hybrid", "forward", True, 64),
    (1, [4, 3, 1], None, False, True, "hybrid", "forward", True, 64),
    (3, [4, 3, 1], None, False, True, "hybrid", "forward", True, 64),
    (5, [4, 3, 1], None, False, True, "hybrid", "forward", True, 64),

    (0, [4, 3, 1], None, True, True, "forward", "forward", True, 64),
    (1, [4, 3, 1], None, True, True, "forward", "forward", True, 64),
    (3, [4, 3, 1], None, True, True, "forward", "forward", True, 64),
    (5, [4, 3, 1], None, True, True, "forward", "forward", True, 64),
    (0, [4, 3, 1], None, False, True, "forward", "forward", True, 64),
    (1, [4, 3, 1], None, False, True, "forward", "forward", True, 64),
    (3, [4, 3, 1], None, False, True, "forward", "forward", True, 64),
    (5, [4, 3, 1], None, False, True, "forward", "forward", True, 64),

    #####
    (0, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 64),
    (1, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 64),
    (3, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 64),
    (5, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 64),
    (0, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 64),
    (1, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 64),
    (3, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 64),
    (5, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 64),

    (0, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 64),
    (1, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 64),
    (3, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 64),
    (5, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 64),
    (0, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 64),
    (1, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 64),
    (3, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 64),
    (5, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 64),

    (0, [4, 3, 1], None, True, True, "forward", "hybrid", True, 64),
    (1, [4, 3, 1], None, True, True, "forward", "hybrid", True, 64),
    (3, [4, 3, 1], None, True, True, "forward", "hybrid", True, 64),
    (5, [4, 3, 1], None, True, True, "forward", "hybrid", True, 64),
    (0, [4, 3, 1], None, False, True, "forward", "hybrid", True, 64),
    (1, [4, 3, 1], None, False, True, "forward", "hybrid", True, 64),
    (3, [4, 3, 1], None, False, True, "forward", "hybrid", True, 64),
    (5, [4, 3, 1], None, False, True, "forward", "hybrid", True, 64),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 64),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 64),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 64),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 64),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 64),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 64),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 64),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 64),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 64),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 64),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 64),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 64),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 64),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 64),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 64),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 64),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 64),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 64),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 64),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 64),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 64),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 64),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 64),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 64),

    (0, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 64),
    (1, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 64),
    (3, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 64),
    (5, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 64),
    (0, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 64),
    (1, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 64),
    (3, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 64),
    (5, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 64),

    ####
    (0, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 64),
    (1, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 64),
    (3, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 64),
    (5, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 64),
    (0, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 64),
    (1, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 64),
    (3, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 64),
    (5, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 64),

    (0, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 64),
    (1, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 64),
    (3, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 64),
    (5, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 64),
    (0, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 64),
    (1, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 64),
    (3, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 64),
    (5, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 64),

    (0, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 64),
    (1, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 64),
    (3, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 64),
    (5, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 64),
    (0, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 64),
    (1, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 64),
    (3, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 64),
    (5, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 64),

])
"""


@pytest.mark.parametrize(
    "n, archi, activation, sequential, use_bias, mode, method, use_input, floatx",
    [
        (0, [4, 3, 1], None, True, True, "ibp", "crown", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", "crown", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", "crown", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", "crown", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", "crown", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", "crown", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", "crown", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", "crown", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", "crown", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", "crown", True, 32),
        (3, [4, 3, 1], None, True, True, "hybrid", "crown", True, 32),
        (5, [4, 3, 1], None, True, True, "hybrid", "crown", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", "crown", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", "crown", True, 32),
        (3, [4, 3, 1], None, False, True, "hybrid", "crown", True, 32),
        (5, [4, 3, 1], None, False, True, "hybrid", "crown", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", "crown", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", "crown", True, 32),
        (3, [4, 3, 1], None, True, True, "forward", "crown", True, 32),
        (5, [4, 3, 1], None, True, True, "forward", "crown", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", "crown", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", "crown", True, 32),
        (3, [4, 3, 1], None, False, True, "forward", "crown", True, 32),
        (5, [4, 3, 1], None, False, True, "forward", "crown", True, 32),
        #####
        (0, [4, 3, 1], None, True, True, "ibp", "ibp", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", "ibp", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", "ibp", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", "ibp", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", "ibp", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", "ibp", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", "ibp", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", "ibp", True, 32),
        #####
        (0, [4, 3, 1], None, True, True, "ibp", "forward", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", "forward", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", "forward", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", "forward", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", "forward", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", "forward", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", "forward", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", "forward", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", "forward", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", "forward", True, 32),
        (3, [4, 3, 1], None, True, True, "hybrid", "forward", True, 32),
        (5, [4, 3, 1], None, True, True, "hybrid", "forward", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", "forward", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", "forward", True, 32),
        (3, [4, 3, 1], None, False, True, "hybrid", "forward", True, 32),
        (5, [4, 3, 1], None, False, True, "hybrid", "forward", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", "forward", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", "forward", True, 32),
        (3, [4, 3, 1], None, True, True, "forward", "forward", True, 32),
        (5, [4, 3, 1], None, True, True, "forward", "forward", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", "forward", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", "forward", True, 32),
        (3, [4, 3, 1], None, False, True, "forward", "forward", True, 32),
        (5, [4, 3, 1], None, False, True, "forward", "forward", True, 32),
        #####
        (0, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", "hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", "hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "hybrid", "hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "hybrid", "hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", "hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", "hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "forward", "hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "forward", "hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", "hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", "hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "forward", "hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "forward", "hybrid", True, 32),
        ####
        (0, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", "crown-ibp", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", "crown-ibp", True, 32),
        ####
        (0, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", "crown-forward", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", "crown-forward", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 32),
        (3, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 32),
        (5, [4, 3, 1], None, True, True, "hybrid", "crown-forward", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 32),
        (3, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 32),
        (5, [4, 3, 1], None, False, True, "hybrid", "crown-forward", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 32),
        (3, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 32),
        (5, [4, 3, 1], None, True, True, "forward", "crown-forward", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 32),
        (3, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 32),
        (5, [4, 3, 1], None, False, True, "forward", "crown-forward", True, 32),
        ####
        (0, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "ibp", "crown-hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "ibp", "crown-hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "hybrid", "crown-hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "hybrid", "crown-hybrid", True, 32),
        (0, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 32),
        (1, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 32),
        (3, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 32),
        (5, [4, 3, 1], None, True, True, "forward", "crown-hybrid", True, 32),
        (0, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 32),
        (1, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 32),
        (3, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 32),
        (5, [4, 3, 1], None, False, True, "forward", "crown-hybrid", True, 32),
    ],
)
def test_convert_1D(n, archi, activation, sequential, use_bias, mode, method, use_input, floatx):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = toy_network_tutorial(dtype=K.floatx())
    ref_nn(inputs[1])

    IBP = True
    forward = True
    if mode == "forward":
        IBP = False
    if mode == "ibp":
        forward = False

    f_dense = clone(ref_nn, method=method, ibp=IBP, forward=forward)

    f_ref = K.function(inputs, ref_nn(inputs[1]))
    y_ref = f_ref(inputs_)

    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(z_)
    if mode == "ibp":
        u_c_, l_c_ = f_dense(z_)
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(z_)

    assert_output_properties_box(
        x_,
        y_ref,
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
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
