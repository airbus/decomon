# Test unit for decomon with Dense layers


import numpy as np
import pytest
import tensorflow.python.keras.backend as K
from numpy.testing import assert_almost_equal
from tensorflow.keras.layers import Dense

from decomon.layers.decomon_layers import DecomonDense, to_monotonic


@pytest.mark.parametrize(
    "n, activation, mode, shared, floatx",
    [
        (0, "relu", "hybrid", False, 32),
        (1, "relu", "hybrid", False, 32),
        (2, "relu", "hybrid", False, 32),
        (3, "relu", "hybrid", False, 32),
        (4, "relu", "hybrid", False, 32),
        (5, "relu", "hybrid", False, 32),
        (6, "relu", "hybrid", False, 32),
        (7, "relu", "hybrid", False, 32),
        (8, "relu", "hybrid", False, 32),
        (9, "relu", "hybrid", False, 32),
        (0, "linear", "hybrid", False, 32),
        (1, "linear", "hybrid", False, 32),
        (2, "linear", "hybrid", False, 32),
        (3, "linear", "hybrid", False, 32),
        (4, "linear", "hybrid", False, 32),
        (5, "linear", "hybrid", False, 32),
        (6, "linear", "hybrid", False, 32),
        (7, "linear", "hybrid", False, 32),
        (8, "linear", "hybrid", False, 32),
        (9, "linear", "hybrid", False, 32),
        (0, None, "hybrid", False, 32),
        (1, None, "hybrid", False, 32),
        (2, None, "hybrid", False, 32),
        (3, None, "hybrid", False, 32),
        (4, None, "hybrid", False, 32),
        (5, None, "hybrid", False, 32),
        (6, None, "hybrid", False, 32),
        (7, None, "hybrid", False, 32),
        (8, None, "hybrid", False, 32),
        (9, None, "hybrid", False, 32),
        (0, "relu", "forward", False, 32),
        (1, "relu", "forward", False, 32),
        (2, "relu", "forward", False, 32),
        (3, "relu", "forward", False, 32),
        (4, "relu", "forward", False, 32),
        (5, "relu", "forward", False, 32),
        (6, "relu", "forward", False, 32),
        (7, "relu", "forward", False, 32),
        (8, "relu", "forward", False, 32),
        (9, "relu", "forward", False, 32),
        (0, "linear", "forward", False, 32),
        (1, "linear", "forward", False, 32),
        (2, "linear", "forward", False, 32),
        (3, "linear", "forward", False, 32),
        (4, "linear", "forward", False, 32),
        (5, "linear", "forward", False, 32),
        (6, "linear", "forward", False, 32),
        (7, "linear", "forward", False, 32),
        (8, "linear", "forward", False, 32),
        (9, "linear", "forward", False, 32),
        (0, None, "forward", False, 32),
        (1, None, "forward", False, 32),
        (2, None, "forward", False, 32),
        (3, None, "forward", False, 32),
        (4, None, "forward", False, 32),
        (5, None, "forward", False, 32),
        (6, None, "forward", False, 32),
        (7, None, "forward", False, 32),
        (8, None, "forward", False, 32),
        (9, None, "forward", False, 32),
        (0, "relu", "ibp", False, 32),
        (1, "relu", "ibp", False, 32),
        (2, "relu", "ibp", False, 32),
        (3, "relu", "ibp", False, 32),
        (4, "relu", "ibp", False, 32),
        (5, "relu", "ibp", False, 32),
        (6, "relu", "ibp", False, 32),
        (7, "relu", "ibp", False, 32),
        (8, "relu", "ibp", False, 32),
        (9, "relu", "ibp", False, 32),
        (0, "linear", "ibp", False, 32),
        (1, "linear", "ibp", False, 32),
        (2, "linear", "ibp", False, 32),
        (3, "linear", "ibp", False, 32),
        (4, "linear", "ibp", False, 32),
        (5, "linear", "ibp", False, 32),
        (6, "linear", "ibp", False, 32),
        (7, "linear", "ibp", False, 32),
        (8, "linear", "ibp", False, 32),
        (9, "linear", "ibp", False, 32),
        (0, None, "ibp", False, 32),
        (1, None, "ibp", False, 32),
        (2, None, "ibp", False, 32),
        (3, None, "ibp", False, 32),
        (4, None, "ibp", False, 32),
        (5, None, "ibp", False, 32),
        (6, None, "ibp", False, 32),
        (7, None, "ibp", False, 32),
        (8, None, "ibp", False, 32),
        (9, None, "ibp", False, 32),
        (0, "relu", "hybrid", True, 32),
        (1, "relu", "hybrid", True, 32),
        (2, "relu", "hybrid", True, 32),
        (3, "relu", "hybrid", True, 32),
        (4, "relu", "hybrid", True, 32),
        (5, "relu", "hybrid", True, 32),
        (6, "relu", "hybrid", True, 32),
        (7, "relu", "hybrid", True, 32),
        (8, "relu", "hybrid", True, 32),
        (9, "relu", "hybrid", True, 32),
        (0, "linear", "hybrid", True, 32),
        (1, "linear", "hybrid", True, 32),
        (2, "linear", "hybrid", True, 32),
        (3, "linear", "hybrid", True, 32),
        (4, "linear", "hybrid", True, 32),
        (5, "linear", "hybrid", True, 32),
        (6, "linear", "hybrid", True, 32),
        (7, "linear", "hybrid", True, 32),
        (8, "linear", "hybrid", True, 32),
        (9, "linear", "hybrid", True, 32),
        (0, None, "hybrid", True, 32),
        (1, None, "hybrid", True, 32),
        (2, None, "hybrid", True, 32),
        (3, None, "hybrid", True, 32),
        (4, None, "hybrid", True, 32),
        (5, None, "hybrid", True, 32),
        (6, None, "hybrid", True, 32),
        (7, None, "hybrid", True, 32),
        (8, None, "hybrid", True, 32),
        (9, None, "hybrid", True, 32),
        (0, "relu", "forward", True, 32),
        (1, "relu", "forward", True, 32),
        (2, "relu", "forward", True, 32),
        (3, "relu", "forward", True, 32),
        (4, "relu", "forward", True, 32),
        (5, "relu", "forward", True, 32),
        (6, "relu", "forward", True, 32),
        (7, "relu", "forward", True, 32),
        (8, "relu", "forward", True, 32),
        (9, "relu", "forward", True, 32),
        (0, "linear", "forward", True, 32),
        (1, "linear", "forward", True, 32),
        (2, "linear", "forward", True, 32),
        (3, "linear", "forward", True, 32),
        (4, "linear", "forward", True, 32),
        (5, "linear", "forward", True, 32),
        (6, "linear", "forward", True, 32),
        (7, "linear", "forward", True, 32),
        (8, "linear", "forward", True, 32),
        (9, "linear", "forward", True, 32),
        (0, None, "forward", True, 32),
        (1, None, "forward", True, 32),
        (2, None, "forward", True, 32),
        (3, None, "forward", True, 32),
        (4, None, "forward", True, 32),
        (5, None, "forward", True, 32),
        (6, None, "forward", True, 32),
        (7, None, "forward", True, 32),
        (8, None, "forward", True, 32),
        (9, None, "forward", True, 32),
        (0, "relu", "ibp", True, 32),
        (1, "relu", "ibp", True, 32),
        (2, "relu", "ibp", True, 32),
        (3, "relu", "ibp", True, 32),
        (4, "relu", "ibp", True, 32),
        (5, "relu", "ibp", True, 32),
        (6, "relu", "ibp", True, 32),
        (7, "relu", "ibp", True, 32),
        (8, "relu", "ibp", True, 32),
        (9, "relu", "ibp", True, 32),
        (0, "linear", "ibp", True, 32),
        (1, "linear", "ibp", True, 32),
        (2, "linear", "ibp", True, 32),
        (3, "linear", "ibp", True, 32),
        (4, "linear", "ibp", True, 32),
        (5, "linear", "ibp", True, 32),
        (6, "linear", "ibp", True, 32),
        (7, "linear", "ibp", True, 32),
        (8, "linear", "ibp", True, 32),
        (9, "linear", "ibp", True, 32),
        (0, None, "ibp", True, 32),
        (1, None, "ibp", True, 32),
        (2, None, "ibp", True, 32),
        (3, None, "ibp", True, 32),
        (4, None, "ibp", True, 32),
        (5, None, "ibp", True, 32),
        (6, None, "ibp", True, 32),
        (7, None, "ibp", True, 32),
        (8, None, "ibp", True, 32),
        (9, None, "ibp", True, 32),
        (0, "relu", "hybrid", False, 64),
        (1, "relu", "hybrid", False, 64),
        (2, "relu", "hybrid", False, 64),
        (3, "relu", "hybrid", False, 64),
        (4, "relu", "hybrid", False, 64),
        (5, "relu", "hybrid", False, 64),
        (6, "relu", "hybrid", False, 64),
        (7, "relu", "hybrid", False, 64),
        (8, "relu", "hybrid", False, 64),
        (9, "relu", "hybrid", False, 64),
        (0, "linear", "hybrid", False, 64),
        (1, "linear", "hybrid", False, 64),
        (2, "linear", "hybrid", False, 64),
        (3, "linear", "hybrid", False, 64),
        (4, "linear", "hybrid", False, 64),
        (5, "linear", "hybrid", False, 64),
        (6, "linear", "hybrid", False, 64),
        (7, "linear", "hybrid", False, 64),
        (8, "linear", "hybrid", False, 64),
        (9, "linear", "hybrid", False, 64),
        (0, None, "hybrid", False, 64),
        (1, None, "hybrid", False, 64),
        (2, None, "hybrid", False, 64),
        (3, None, "hybrid", False, 64),
        (4, None, "hybrid", False, 64),
        (5, None, "hybrid", False, 64),
        (6, None, "hybrid", False, 64),
        (7, None, "hybrid", False, 64),
        (8, None, "hybrid", False, 64),
        (9, None, "hybrid", False, 64),
        (0, "relu", "forward", False, 64),
        (1, "relu", "forward", False, 64),
        (2, "relu", "forward", False, 64),
        (3, "relu", "forward", False, 64),
        (4, "relu", "forward", False, 64),
        (5, "relu", "forward", False, 64),
        (6, "relu", "forward", False, 64),
        (7, "relu", "forward", False, 64),
        (8, "relu", "forward", False, 64),
        (9, "relu", "forward", False, 64),
        (0, "linear", "forward", False, 64),
        (1, "linear", "forward", False, 64),
        (2, "linear", "forward", False, 64),
        (3, "linear", "forward", False, 64),
        (4, "linear", "forward", False, 64),
        (5, "linear", "forward", False, 64),
        (6, "linear", "forward", False, 64),
        (7, "linear", "forward", False, 64),
        (8, "linear", "forward", False, 64),
        (9, "linear", "forward", False, 64),
        (0, None, "forward", False, 64),
        (1, None, "forward", False, 64),
        (2, None, "forward", False, 64),
        (3, None, "forward", False, 64),
        (4, None, "forward", False, 64),
        (5, None, "forward", False, 64),
        (6, None, "forward", False, 64),
        (7, None, "forward", False, 64),
        (8, None, "forward", False, 64),
        (9, None, "forward", False, 64),
        (0, "relu", "ibp", False, 64),
        (1, "relu", "ibp", False, 64),
        (2, "relu", "ibp", False, 64),
        (3, "relu", "ibp", False, 64),
        (4, "relu", "ibp", False, 64),
        (5, "relu", "ibp", False, 64),
        (6, "relu", "ibp", False, 64),
        (7, "relu", "ibp", False, 64),
        (8, "relu", "ibp", False, 64),
        (9, "relu", "ibp", False, 64),
        (0, "linear", "ibp", False, 64),
        (1, "linear", "ibp", False, 64),
        (2, "linear", "ibp", False, 64),
        (3, "linear", "ibp", False, 64),
        (4, "linear", "ibp", False, 64),
        (5, "linear", "ibp", False, 64),
        (6, "linear", "ibp", False, 64),
        (7, "linear", "ibp", False, 64),
        (8, "linear", "ibp", False, 64),
        (9, "linear", "ibp", False, 64),
        (0, None, "ibp", False, 64),
        (1, None, "ibp", False, 64),
        (2, None, "ibp", False, 64),
        (3, None, "ibp", False, 64),
        (4, None, "ibp", False, 64),
        (5, None, "ibp", False, 64),
        (6, None, "ibp", False, 64),
        (7, None, "ibp", False, 64),
        (8, None, "ibp", False, 64),
        (9, None, "ibp", False, 64),
        (0, "relu", "hybrid", True, 64),
        (1, "relu", "hybrid", True, 64),
        (2, "relu", "hybrid", True, 64),
        (3, "relu", "hybrid", True, 64),
        (4, "relu", "hybrid", True, 64),
        (5, "relu", "hybrid", True, 64),
        (6, "relu", "hybrid", True, 64),
        (7, "relu", "hybrid", True, 64),
        (8, "relu", "hybrid", True, 64),
        (9, "relu", "hybrid", True, 64),
        (0, "linear", "hybrid", True, 64),
        (1, "linear", "hybrid", True, 64),
        (2, "linear", "hybrid", True, 64),
        (3, "linear", "hybrid", True, 64),
        (4, "linear", "hybrid", True, 64),
        (5, "linear", "hybrid", True, 64),
        (6, "linear", "hybrid", True, 64),
        (7, "linear", "hybrid", True, 64),
        (8, "linear", "hybrid", True, 64),
        (9, "linear", "hybrid", True, 64),
        (0, None, "hybrid", True, 64),
        (1, None, "hybrid", True, 64),
        (2, None, "hybrid", True, 64),
        (3, None, "hybrid", True, 64),
        (4, None, "hybrid", True, 64),
        (5, None, "hybrid", True, 64),
        (6, None, "hybrid", True, 64),
        (7, None, "hybrid", True, 64),
        (8, None, "hybrid", True, 64),
        (9, None, "hybrid", True, 64),
        (0, "relu", "forward", True, 64),
        (1, "relu", "forward", True, 64),
        (2, "relu", "forward", True, 64),
        (3, "relu", "forward", True, 64),
        (4, "relu", "forward", True, 64),
        (5, "relu", "forward", True, 64),
        (6, "relu", "forward", True, 64),
        (7, "relu", "forward", True, 64),
        (8, "relu", "forward", True, 64),
        (9, "relu", "forward", True, 64),
        (0, "linear", "forward", True, 64),
        (1, "linear", "forward", True, 64),
        (2, "linear", "forward", True, 64),
        (3, "linear", "forward", True, 64),
        (4, "linear", "forward", True, 64),
        (5, "linear", "forward", True, 64),
        (6, "linear", "forward", True, 64),
        (7, "linear", "forward", True, 64),
        (8, "linear", "forward", True, 64),
        (9, "linear", "forward", True, 64),
        (0, None, "forward", True, 64),
        (1, None, "forward", True, 64),
        (2, None, "forward", True, 64),
        (3, None, "forward", True, 64),
        (4, None, "forward", True, 64),
        (5, None, "forward", True, 64),
        (6, None, "forward", True, 64),
        (7, None, "forward", True, 64),
        (8, None, "forward", True, 64),
        (9, None, "forward", True, 64),
        (0, "relu", "ibp", True, 64),
        (1, "relu", "ibp", True, 64),
        (2, "relu", "ibp", True, 64),
        (3, "relu", "ibp", True, 64),
        (4, "relu", "ibp", True, 64),
        (5, "relu", "ibp", True, 64),
        (6, "relu", "ibp", True, 64),
        (7, "relu", "ibp", True, 64),
        (8, "relu", "ibp", True, 64),
        (9, "relu", "ibp", True, 64),
        (0, "linear", "ibp", True, 64),
        (1, "linear", "ibp", True, 64),
        (2, "linear", "ibp", True, 64),
        (3, "linear", "ibp", True, 64),
        (4, "linear", "ibp", True, 64),
        (5, "linear", "ibp", True, 64),
        (6, "linear", "ibp", True, 64),
        (7, "linear", "ibp", True, 64),
        (8, "linear", "ibp", True, 64),
        (9, "linear", "ibp", True, 64),
        (0, None, "ibp", True, 64),
        (1, None, "ibp", True, 64),
        (2, None, "ibp", True, 64),
        (3, None, "ibp", True, 64),
        (4, None, "ibp", True, 64),
        (5, None, "ibp", True, 64),
        (6, None, "ibp", True, 64),
        (7, None, "ibp", True, 64),
        (8, None, "ibp", True, 64),
        (9, None, "ibp", True, 64),
        (0, "relu", "hybrid", False, 16),
        (1, "relu", "hybrid", False, 16),
        (2, "relu", "hybrid", False, 16),
        (3, "relu", "hybrid", False, 16),
        (4, "relu", "hybrid", False, 16),
        (5, "relu", "hybrid", False, 16),
        (6, "relu", "hybrid", False, 16),
        (7, "relu", "hybrid", False, 16),
        (8, "relu", "hybrid", False, 16),
        (9, "relu", "hybrid", False, 16),
        (0, "linear", "hybrid", False, 16),
        (1, "linear", "hybrid", False, 16),
        (2, "linear", "hybrid", False, 16),
        (3, "linear", "hybrid", False, 16),
        (4, "linear", "hybrid", False, 16),
        (5, "linear", "hybrid", False, 16),
        (6, "linear", "hybrid", False, 16),
        (7, "linear", "hybrid", False, 16),
        (8, "linear", "hybrid", False, 16),
        (9, "linear", "hybrid", False, 16),
        (0, None, "hybrid", False, 16),
        (1, None, "hybrid", False, 16),
        (2, None, "hybrid", False, 16),
        (3, None, "hybrid", False, 16),
        (4, None, "hybrid", False, 16),
        (5, None, "hybrid", False, 16),
        (6, None, "hybrid", False, 16),
        (7, None, "hybrid", False, 16),
        (8, None, "hybrid", False, 16),
        (9, None, "hybrid", False, 16),
        (0, "relu", "forward", False, 16),
        (1, "relu", "forward", False, 16),
        (2, "relu", "forward", False, 16),
        (3, "relu", "forward", False, 16),
        (4, "relu", "forward", False, 16),
        (5, "relu", "forward", False, 16),
        (6, "relu", "forward", False, 16),
        (7, "relu", "forward", False, 16),
        (8, "relu", "forward", False, 16),
        (9, "relu", "forward", False, 16),
        (0, "linear", "forward", False, 16),
        (1, "linear", "forward", False, 16),
        (2, "linear", "forward", False, 16),
        (3, "linear", "forward", False, 16),
        (4, "linear", "forward", False, 16),
        (5, "linear", "forward", False, 16),
        (6, "linear", "forward", False, 16),
        (7, "linear", "forward", False, 16),
        (8, "linear", "forward", False, 16),
        (9, "linear", "forward", False, 16),
        (0, None, "forward", False, 16),
        (1, None, "forward", False, 16),
        (2, None, "forward", False, 16),
        (3, None, "forward", False, 16),
        (4, None, "forward", False, 16),
        (5, None, "forward", False, 16),
        (6, None, "forward", False, 16),
        (7, None, "forward", False, 16),
        (8, None, "forward", False, 16),
        (9, None, "forward", False, 16),
        (0, "relu", "ibp", False, 16),
        (1, "relu", "ibp", False, 16),
        (2, "relu", "ibp", False, 16),
        (3, "relu", "ibp", False, 16),
        (4, "relu", "ibp", False, 16),
        (5, "relu", "ibp", False, 16),
        (6, "relu", "ibp", False, 16),
        (7, "relu", "ibp", False, 16),
        (8, "relu", "ibp", False, 16),
        (9, "relu", "ibp", False, 16),
        (0, "linear", "ibp", False, 16),
        (1, "linear", "ibp", False, 16),
        (2, "linear", "ibp", False, 16),
        (3, "linear", "ibp", False, 16),
        (4, "linear", "ibp", False, 16),
        (5, "linear", "ibp", False, 16),
        (6, "linear", "ibp", False, 16),
        (7, "linear", "ibp", False, 16),
        (8, "linear", "ibp", False, 16),
        (9, "linear", "ibp", False, 16),
        (0, None, "ibp", False, 16),
        (1, None, "ibp", False, 16),
        (2, None, "ibp", False, 16),
        (3, None, "ibp", False, 16),
        (4, None, "ibp", False, 16),
        (5, None, "ibp", False, 16),
        (6, None, "ibp", False, 16),
        (7, None, "ibp", False, 16),
        (8, None, "ibp", False, 16),
        (9, None, "ibp", False, 16),
        (0, "relu", "hybrid", True, 16),
        (1, "relu", "hybrid", True, 16),
        (2, "relu", "hybrid", True, 16),
        (3, "relu", "hybrid", True, 16),
        (4, "relu", "hybrid", True, 16),
        (5, "relu", "hybrid", True, 16),
        (6, "relu", "hybrid", True, 16),
        (7, "relu", "hybrid", True, 16),
        (8, "relu", "hybrid", True, 16),
        (9, "relu", "hybrid", True, 16),
        (0, "linear", "hybrid", True, 16),
        (1, "linear", "hybrid", True, 16),
        (2, "linear", "hybrid", True, 16),
        (3, "linear", "hybrid", True, 16),
        (4, "linear", "hybrid", True, 16),
        (5, "linear", "hybrid", True, 16),
        (6, "linear", "hybrid", True, 16),
        (7, "linear", "hybrid", True, 16),
        (8, "linear", "hybrid", True, 16),
        (9, "linear", "hybrid", True, 16),
        (0, None, "hybrid", True, 16),
        (1, None, "hybrid", True, 16),
        (2, None, "hybrid", True, 16),
        (3, None, "hybrid", True, 16),
        (4, None, "hybrid", True, 16),
        (5, None, "hybrid", True, 16),
        (6, None, "hybrid", True, 16),
        (7, None, "hybrid", True, 16),
        (8, None, "hybrid", True, 16),
        (9, None, "hybrid", True, 16),
        (0, "relu", "forward", True, 16),
        (1, "relu", "forward", True, 16),
        (2, "relu", "forward", True, 16),
        (3, "relu", "forward", True, 16),
        (4, "relu", "forward", True, 16),
        (5, "relu", "forward", True, 16),
        (6, "relu", "forward", True, 16),
        (7, "relu", "forward", True, 16),
        (8, "relu", "forward", True, 16),
        (9, "relu", "forward", True, 16),
        (0, "linear", "forward", True, 16),
        (1, "linear", "forward", True, 16),
        (2, "linear", "forward", True, 16),
        (3, "linear", "forward", True, 16),
        (4, "linear", "forward", True, 16),
        (5, "linear", "forward", True, 16),
        (6, "linear", "forward", True, 16),
        (7, "linear", "forward", True, 16),
        (8, "linear", "forward", True, 16),
        (9, "linear", "forward", True, 16),
        (0, None, "forward", True, 16),
        (1, None, "forward", True, 16),
        (2, None, "forward", True, 16),
        (3, None, "forward", True, 16),
        (4, None, "forward", True, 16),
        (5, None, "forward", True, 16),
        (6, None, "forward", True, 16),
        (7, None, "forward", True, 16),
        (8, None, "forward", True, 16),
        (9, None, "forward", True, 16),
        (0, "relu", "ibp", True, 16),
        (1, "relu", "ibp", True, 16),
        (2, "relu", "ibp", True, 16),
        (3, "relu", "ibp", True, 16),
        (4, "relu", "ibp", True, 16),
        (5, "relu", "ibp", True, 16),
        (6, "relu", "ibp", True, 16),
        (7, "relu", "ibp", True, 16),
        (8, "relu", "ibp", True, 16),
        (9, "relu", "ibp", True, 16),
        (0, "linear", "ibp", True, 16),
        (1, "linear", "ibp", True, 16),
        (2, "linear", "ibp", True, 16),
        (3, "linear", "ibp", True, 16),
        (4, "linear", "ibp", True, 16),
        (5, "linear", "ibp", True, 16),
        (6, "linear", "ibp", True, 16),
        (7, "linear", "ibp", True, 16),
        (8, "linear", "ibp", True, 16),
        (9, "linear", "ibp", True, 16),
        (0, None, "ibp", True, 16),
        (1, None, "ibp", True, 16),
        (2, None, "ibp", True, 16),
        (3, None, "ibp", True, 16),
        (4, None, "ibp", True, 16),
        (5, None, "ibp", True, 16),
        (6, None, "ibp", True, 16),
        (7, None, "ibp", True, 16),
        (8, None, "ibp", True, 16),
        (9, None, "ibp", True, 16),
    ],
)
def test_DecomonDense_1D_box(n, activation, mode, shared, floatx, helpers):

    K.set_floatx(f"float{floatx}")
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    monotonic_dense = DecomonDense(
        1, use_bias=True, activation=activation, dc_decomp=True, mode=mode, shared=shared, dtype=K.floatx()
    )

    ref_dense = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standart_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_dense(inputs[1])
    monotonic_dense.shared_weights(ref_dense)

    if mode == "hybrid":
        output = monotonic_dense(inputs[2:])
    if mode == "forward":
        output = monotonic_dense([z, W_u, b_u, W_l, b_l, h, g])
    if mode == "ibp":
        output = monotonic_dense([u_c, l_c, h, g])

    W_, bias = monotonic_dense.get_weights()
    if not shared:
        monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    y_ref = f_ref(inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
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
            f"dense_{n}",
            decimal=decimal,
        )

    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        u_c_ = None
        l_c_ = None
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
            f"dense_{n}",
            decimal=decimal,
        )

    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            f"dense_{n}",
            decimal=decimal,
        )
    if not shared:
        monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    y_ref = f_ref(inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
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
            f"dense_{n}",
            decimal=decimal,
        )

    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        u_c_ = None
        l_c_ = None
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
            f"dense_{n}",
            decimal=decimal,
        )

    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            f"dense_{n}",
            decimal=decimal,
        )

    K.set_epsilon(eps)
    K.set_floatx("float32")


@pytest.mark.parametrize(
    "odd, activation, mode",
    [
        (0, None, "hybrid"),
        (1, None, "hybrid"),
        (0, "linear", "hybrid"),
        (1, "linear", "hybrid"),
        (0, "relu", "hybrid"),
        (1, "relu", "hybrid"),
        (0, None, "forward"),
        (1, None, "forward"),
        (0, "linear", "forward"),
        (1, "linear", "forward"),
        (0, "relu", "forward"),
        (1, "relu", "forward"),
        (0, None, "ibp"),
        (1, None, "ibp"),
        (0, "linear", "ibp"),
        (1, "linear", "ibp"),
        (0, "relu", "ibp"),
        (1, "relu", "ibp"),
    ],
)
def test_DecomonDense_multiD_box(odd, activation, mode, helpers):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=True, mode=mode, dtype=K.floatx())
    ref_dense = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_multid_box(odd)
    inputs_ = helpers.get_standard_values_multid_box(odd)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]
    if mode == "hybrid":
        output = monotonic_dense(inputs[2:])
    if mode == "forward":
        output = monotonic_dense([z, W_u, b_u, W_l, b_l, h, g])
    if mode == "ibp":
        output = monotonic_dense([u_c, l_c, h, g])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))
    y_ref = f_ref(inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
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
            f"dense_multid_{odd}",
        )
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        u_c_ = None
        l_c_ = None
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
            f"dense_multid_{odd}",
        )
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "dense_multid_{}".format(odd),
        )

    monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    y_ref = f_ref(inputs_)

    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
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
            "dense_multid_{}".format(odd),
        )
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        u_c_ = None
        l_c_ = None
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
            "dense_multid_{}".format(odd),
        )
    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "dense_multid_{}".format(odd),
        )


@pytest.mark.parametrize(
    "n, activation, mode, shared",
    [
        (0, "relu", "hybrid", False),
        (1, "relu", "hybrid", False),
        (2, "relu", "hybrid", False),
        (3, "relu", "hybrid", False),
        (4, "relu", "hybrid", False),
        (5, "relu", "hybrid", False),
        (6, "relu", "hybrid", False),
        (7, "relu", "hybrid", False),
        (8, "relu", "hybrid", False),
        (9, "relu", "hybrid", False),
        (0, "linear", "hybrid", False),
        (1, "linear", "hybrid", False),
        (2, "linear", "hybrid", False),
        (3, "linear", "hybrid", False),
        (4, "linear", "hybrid", False),
        (5, "linear", "hybrid", False),
        (6, "linear", "hybrid", False),
        (7, "linear", "hybrid", False),
        (8, "linear", "hybrid", False),
        (9, "linear", "hybrid", False),
        (0, None, "hybrid", False),
        (1, None, "hybrid", False),
        (2, None, "hybrid", False),
        (3, None, "hybrid", False),
        (4, None, "hybrid", False),
        (5, None, "hybrid", False),
        (6, None, "hybrid", False),
        (7, None, "hybrid", False),
        (8, None, "hybrid", False),
        (9, None, "hybrid", False),
        (0, "relu", "forward", False),
        (1, "relu", "forward", False),
        (2, "relu", "forward", False),
        (3, "relu", "forward", False),
        (4, "relu", "forward", False),
        (5, "relu", "forward", False),
        (6, "relu", "forward", False),
        (7, "relu", "forward", False),
        (8, "relu", "forward", False),
        (9, "relu", "forward", False),
        (0, "linear", "forward", False),
        (1, "linear", "forward", False),
        (2, "linear", "forward", False),
        (3, "linear", "forward", False),
        (4, "linear", "forward", False),
        (5, "linear", "forward", False),
        (6, "linear", "forward", False),
        (7, "linear", "forward", False),
        (8, "linear", "forward", False),
        (9, "linear", "forward", False),
        (0, None, "forward", False),
        (1, None, "forward", False),
        (2, None, "forward", False),
        (3, None, "forward", False),
        (4, None, "forward", False),
        (5, None, "forward", False),
        (6, None, "forward", False),
        (7, None, "forward", False),
        (8, None, "forward", False),
        (9, None, "forward", False),
        (0, "relu", "ibp", False),
        (1, "relu", "ibp", False),
        (2, "relu", "ibp", False),
        (3, "relu", "ibp", False),
        (4, "relu", "ibp", False),
        (5, "relu", "ibp", False),
        (6, "relu", "ibp", False),
        (7, "relu", "ibp", False),
        (8, "relu", "ibp", False),
        (9, "relu", "ibp", False),
        (0, "linear", "ibp", False),
        (1, "linear", "ibp", False),
        (2, "linear", "ibp", False),
        (3, "linear", "ibp", False),
        (4, "linear", "ibp", False),
        (5, "linear", "ibp", False),
        (6, "linear", "ibp", False),
        (7, "linear", "ibp", False),
        (8, "linear", "ibp", False),
        (9, "linear", "ibp", False),
        (0, None, "ibp", False),
        (1, None, "ibp", False),
        (2, None, "ibp", False),
        (3, None, "ibp", False),
        (4, None, "ibp", False),
        (5, None, "ibp", False),
        (6, None, "ibp", False),
        (7, None, "ibp", False),
        (8, None, "ibp", False),
        (9, None, "ibp", False),
        (0, "relu", "hybrid", True),
        (1, "relu", "hybrid", True),
        (2, "relu", "hybrid", True),
        (3, "relu", "hybrid", True),
        (4, "relu", "hybrid", True),
        (5, "relu", "hybrid", True),
        (6, "relu", "hybrid", True),
        (7, "relu", "hybrid", True),
        (8, "relu", "hybrid", True),
        (9, "relu", "hybrid", True),
        (0, "linear", "hybrid", True),
        (1, "linear", "hybrid", True),
        (2, "linear", "hybrid", True),
        (3, "linear", "hybrid", True),
        (4, "linear", "hybrid", True),
        (5, "linear", "hybrid", True),
        (6, "linear", "hybrid", True),
        (7, "linear", "hybrid", True),
        (8, "linear", "hybrid", True),
        (9, "linear", "hybrid", True),
        (0, None, "hybrid", True),
        (1, None, "hybrid", True),
        (2, None, "hybrid", True),
        (3, None, "hybrid", True),
        (4, None, "hybrid", True),
        (5, None, "hybrid", True),
        (6, None, "hybrid", True),
        (7, None, "hybrid", True),
        (8, None, "hybrid", True),
        (9, None, "hybrid", True),
        (0, "relu", "forward", True),
        (1, "relu", "forward", True),
        (2, "relu", "forward", True),
        (3, "relu", "forward", True),
        (4, "relu", "forward", True),
        (5, "relu", "forward", True),
        (6, "relu", "forward", True),
        (7, "relu", "forward", True),
        (8, "relu", "forward", True),
        (9, "relu", "forward", True),
        (0, "linear", "forward", True),
        (1, "linear", "forward", True),
        (2, "linear", "forward", True),
        (3, "linear", "forward", True),
        (4, "linear", "forward", True),
        (5, "linear", "forward", True),
        (6, "linear", "forward", True),
        (7, "linear", "forward", True),
        (8, "linear", "forward", True),
        (9, "linear", "forward", True),
        (0, None, "forward", True),
        (1, None, "forward", True),
        (2, None, "forward", True),
        (3, None, "forward", True),
        (4, None, "forward", True),
        (5, None, "forward", True),
        (6, None, "forward", True),
        (7, None, "forward", True),
        (8, None, "forward", True),
        (9, None, "forward", True),
        (0, "relu", "ibp", True),
        (1, "relu", "ibp", True),
        (2, "relu", "ibp", True),
        (3, "relu", "ibp", True),
        (4, "relu", "ibp", True),
        (5, "relu", "ibp", True),
        (6, "relu", "ibp", True),
        (7, "relu", "ibp", True),
        (8, "relu", "ibp", True),
        (9, "relu", "ibp", True),
        (0, "linear", "ibp", True),
        (1, "linear", "ibp", True),
        (2, "linear", "ibp", True),
        (3, "linear", "ibp", True),
        (4, "linear", "ibp", True),
        (5, "linear", "ibp", True),
        (6, "linear", "ibp", True),
        (7, "linear", "ibp", True),
        (8, "linear", "ibp", True),
        (9, "linear", "ibp", True),
        (0, None, "ibp", True),
        (1, None, "ibp", True),
        (2, None, "ibp", True),
        (3, None, "ibp", True),
        (4, None, "ibp", True),
        (5, None, "ibp", True),
        (6, None, "ibp", True),
        (7, None, "ibp", True),
        (8, None, "ibp", True),
        (9, None, "ibp", True),
    ],
)
def test_DecomonDense_1D_to_monotonic_box(n, activation, mode, shared, helpers):

    dense_ref = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_1d_box()
    inputs_ = helpers.get_standart_values_1d_box(n)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    z_ = inputs_[2]
    x_ = inputs_[0]

    output_ref = dense_ref(inputs[1])
    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]
    if mode == "hybrid":
        IBP = True
        forward = True
    if mode == "forward":
        IBP = False
        forward = True
    if mode == "ibp":
        IBP = True
        forward = False
    monotonic_dense = to_monotonic(
        dense_ref, (2, input_dim), dc_decomp=True, IBP=IBP, forward=forward, shared=shared
    )  # ATTENTION: it will be a list

    W_, bias = monotonic_dense[0].get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    if mode == "hybrid":
        output = monotonic_dense[0](inputs[2:])
    if mode == "forward":
        output = monotonic_dense[0]([z, W_u, b_u, W_l, b_l, h, g])
    if mode == "ibp":
        output = monotonic_dense[0]([u_c, l_c, h, g])

    if len(monotonic_dense) > 1:
        output = monotonic_dense[1](output)

    f_dense = K.function(inputs[2:], output)
    y_ref = f_ref(inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
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
            "dense_to_monotonic_{}".format(n),
            decimal=5,
        )

    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            None,
            w_u_,
            b_u_,
            None,
            w_l_,
            b_l_,
            "dense_to_monotonic_{}".format(n),
            decimal=5,
        )

    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "dense_to_monotonic_{}".format(n),
            decimal=5,
        )


@pytest.mark.parametrize(
    "odd, activation, mode",
    [
        (0, None, "hybrid"),
        (1, None, "hybrid"),
        (0, "linear", "hybrid"),
        (1, "linear", "hybrid"),
        (0, "relu", "hybrid"),
        (1, "relu", "hybrid"),
        (0, None, "forward"),
        (1, None, "forward"),
        (0, "linear", "forward"),
        (1, "linear", "forward"),
        (0, "relu", "forward"),
        (1, "relu", "forward"),
        (0, None, "ibp"),
        (1, None, "ibp"),
        (0, "linear", "ibp"),
        (1, "linear", "ibp"),
        (0, "relu", "ibp"),
        (1, "relu", "ibp"),
    ],
)
def test_DecomonDense_multiD_to_monotonic_box(odd, activation, mode, helpers):

    dense_ref = Dense(1, use_bias=True, activation=activation, dtype=K.floatx())

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=True)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=True)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    output_ref = dense_ref(inputs[1])
    W_0, b_0 = dense_ref.get_weights()
    if odd == 1:

        W_0[0] = -0.1657672
        W_0[1] = -0.2613032
        W_0[2] = 0.08437371
        dense_ref.set_weights([W_0, b_0])

    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]

    y_ = f_ref(inputs_)

    if mode == "hybrid":
        IBP = True
        forward = True
    if mode == "forward":
        IBP = False
        forward = True
    if mode == "ibp":
        IBP = True
        forward = False

    monotonic_dense = to_monotonic(dense_ref, (2, input_dim), dc_decomp=True, IBP=IBP, forward=forward)

    W_, bias = monotonic_dense[0].get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    if mode == "hybrid":
        output = monotonic_dense[0](inputs[2:])
    if mode == "forward":
        output = monotonic_dense[0]([z, W_u, b_u, W_l, b_l, h, g])
    if mode == "ibp":
        output = monotonic_dense[0]([u_c, l_c, h, g])

    if len(monotonic_dense) > 1:
        output = monotonic_dense[1](output)

    f_dense = K.function(inputs[2:], output)

    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
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
            "dense_to_monotonic_{}".format(0),
            decimal=5,
        )

    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            None,
            w_u_,
            b_u_,
            None,
            w_l_,
            b_l_,
            "dense_to_monotonic_{}".format(0),
            decimal=5,
        )

    if mode == "ibp":
        u_c_, l_c_, h_, g_ = f_dense(inputs_[2:])
        helpers.assert_output_properties_box(
            x_,
            y_,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "dense_to_monotonic_{}".format(0),
            decimal=5,
        )


"""

# DC DECOMP = FALSE
"""


@pytest.mark.parametrize(
    "n, activation, n_subgrad",
    [
        (0, "relu", 0),
        (1, "relu", 0),
        (2, "relu", 0),
        (3, "relu", 0),
        (4, "relu", 0),
        (5, "relu", 0),
        (6, "relu", 0),
        (7, "relu", 0),
        (8, "relu", 0),
        (9, "relu", 0),
        (0, "linear", 0),
        (1, "linear", 0),
        (2, "linear", 0),
        (3, "linear", 0),
        (4, "linear", 0),
        (5, "linear", 0),
        (6, "linear", 0),
        (7, "linear", 0),
        (8, "linear", 0),
        (9, "linear", 0),
        (0, None, 0),
        (1, None, 0),
        (2, None, 0),
        (3, None, 0),
        (4, None, 0),
        (5, None, 0),
        (6, None, 0),
        (7, None, 0),
        (8, None, 0),
        (9, None, 0),
        (0, "relu", 1),
        (1, "relu", 1),
        (2, "relu", 1),
        (3, "relu", 1),
        (4, "relu", 1),
        (5, "relu", 1),
        (6, "relu", 1),
        (7, "relu", 1),
        (8, "relu", 1),
        (9, "relu", 1),
        (0, "linear", 1),
        (1, "linear", 1),
        (2, "linear", 1),
        (3, "linear", 1),
        (4, "linear", 1),
        (5, "linear", 1),
        (6, "linear", 1),
        (7, "linear", 1),
        (8, "linear", 1),
        (9, "linear", 1),
        (0, None, 1),
        (1, None, 1),
        (2, None, 1),
        (3, None, 1),
        (4, None, 1),
        (5, None, 1),
        (6, None, 1),
        (7, None, 1),
        (8, None, 1),
        (9, None, 1),
        (0, "relu", 5),
        (1, "relu", 5),
        (2, "relu", 5),
        (3, "relu", 5),
        (4, "relu", 5),
        (5, "relu", 5),
        (6, "relu", 5),
        (7, "relu", 5),
        (8, "relu", 5),
        (9, "relu", 5),
        (0, "linear", 5),
        (1, "linear", 5),
        (2, "linear", 5),
        (3, "linear", 5),
        (4, "linear", 5),
        (5, "linear", 5),
        (6, "linear", 5),
        (7, "linear", 5),
        (8, "linear", 5),
        (9, "linear", 5),
        (0, None, 5),
        (1, None, 5),
        (2, None, 5),
        (3, None, 5),
        (4, None, 5),
        (5, None, 5),
        (6, None, 5),
        (7, None, 5),
        (8, None, 5),
        (9, None, 5),
    ],
)
def test_DecomonDense_1D_box_nodc(n, activation, n_subgrad, helpers):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_dense(inputs[2:])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])

    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    y_ref = f_ref(inputs_)
    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )


@pytest.mark.parametrize(
    "odd, activation, mode",
    [
        (0, None, "hybrid"),
        (1, None, "hybrid"),
        (0, "linear", "hybrid"),
        (1, "linear", "hybrid"),
        (0, "relu", "hybrid"),
        (1, "relu", "hybrid"),
        (0, None, "forward"),
        (1, None, "forward"),
        (0, "linear", "forward"),
        (1, "linear", "forward"),
        (0, "relu", "forward"),
        (1, "relu", "forward"),
        (0, None, "ibp"),
        (1, None, "ibp"),
        (0, "linear", "ibp"),
        (1, "linear", "ibp"),
        (0, "relu", "ibp"),
        (1, "relu", "ibp"),
    ],
)
def test_DecomonDense_multiD_to_monotonic_box_nodc(odd, activation, mode, helpers):

    dense_ref = Dense(1, use_bias=True, activation=activation)

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output_ref = dense_ref(inputs[1])
    W_0, b_0 = dense_ref.get_weights()
    if odd == 0:
        W_0[0] = 1.037377
        W_0[1] = -0.7575816

        dense_ref.set_weights([W_0, b_0])
    if odd == 1:

        W_0[0] = -0.1657672
        W_0[1] = -0.2613032
        W_0[2] = 0.08437371
        dense_ref.set_weights([W_0, b_0])

    f_ref = K.function(inputs, output_ref)
    input_dim = x.shape[-1]

    y_ref = f_ref(inputs_)

    monotonic_dense = to_monotonic(dense_ref, (2, input_dim), dc_decomp=False)

    W_, bias = monotonic_dense[0].get_weights()
    W_0, b_0 = dense_ref.get_weights()

    assert_almost_equal(W_, W_0, decimal=6, err_msg="wrong decomposition")
    assert_almost_equal(bias, b_0, decimal=6, err_msg="wrong decomposition")

    output = monotonic_dense[0](inputs[2:])
    if len(monotonic_dense) > 1:
        output = monotonic_dense[1](output)

    f_dense = K.function(inputs[2:], output)

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])

    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )


@pytest.mark.parametrize(
    "odd, activation, n_subgrad",
    [
        (0, None, 0),
        (1, None, 0),
        (0, "linear", 0),
        (1, "linear", 0),
        (0, "relu", 0),
        (1, "relu", 0),
        (0, None, 1),
        (1, None, 1),
        (0, "linear", 1),
        (1, "linear", 1),
        (0, "relu", 1),
        (1, "relu", 1),
        (0, None, 5),
        (1, None, 5),
        (0, "linear", 5),
        (1, "linear", 5),
        (0, "relu", 5),
        (1, "relu", 5),
    ],
)
def test_DecomonDense_multiD_box_dc(odd, activation, n_subgrad, helpers):

    monotonic_dense = DecomonDense(1, use_bias=True, activation=activation, dc_decomp=False)
    ref_dense = Dense(1, use_bias=True, activation=activation)

    inputs = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output = monotonic_dense(inputs[2:])
    ref_dense(inputs[1])

    W_, bias = monotonic_dense.get_weights()

    monotonic_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([2 * np.ones_like(W_), np.ones_like(bias)])
    f_dense = K.function(inputs[2:], output)
    f_ref = K.function(inputs, ref_dense(inputs[1]))

    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    y_ref = f_ref(inputs_)

    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )

    monotonic_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    ref_dense.set_weights([-3 * np.ones_like(W_), np.ones_like(bias)])
    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(inputs_[2:])
    y_ref = f_ref(inputs_)

    helpers.assert_output_properties_box_linear(
        x, y_ref, z_[:, 0], z_[:, 1], u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, "nodc"
    )
