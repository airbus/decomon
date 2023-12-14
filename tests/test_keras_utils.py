import keras
import keras.ops as K
import numpy as np
import pytest
from keras.layers import Dense, Input
from numpy.testing import assert_almost_equal

from decomon.keras_utils import (
    BACKEND_PYTORCH,
    BACKEND_TENSORFLOW,
    LinalgSolve,
    get_weight_index_from_name,
    share_layer_all_weights,
)


def test_get_weight_index_from_name_nok_attribute():
    layer = Dense(3)
    layer(K.zeros((2, 1)))
    with pytest.raises(AttributeError):
        get_weight_index_from_name(layer=layer, weight_name="toto")


def test_get_weight_index_from_name_nok_index():
    layer = Dense(3, use_bias=False)
    layer(K.zeros((2, 1)))
    with pytest.raises(IndexError):
        get_weight_index_from_name(layer=layer, weight_name="bias")


def test_get_weight_index_from_name_ok():
    layer = Dense(3)
    layer(K.zeros((2, 1)))
    assert get_weight_index_from_name(layer=layer, weight_name="bias") in [0, 1]


def test_linalgsolve(floatx, decimal):
    if keras.config.backend() in (BACKEND_TENSORFLOW, BACKEND_PYTORCH) and floatx == 16:
        pytest.skip("LinalgSolve not implemented for float16 on torch and tensorflow")

    dtype = f"float{floatx}"

    matrix = np.array([[1, 0, 0], [2, 1, 0], [3, 2, 1]])
    matrix = np.repeat(matrix[None, None], 2, axis=0)
    matrix_symbolic_tensor = keras.KerasTensor(shape=matrix.shape, dtype=dtype)
    matrix_tensor = keras.ops.convert_to_tensor(matrix, dtype=dtype)

    rhs = np.array([[1, 0], [0, 0], [0, 1]])
    rhs = np.repeat(rhs[None, None], 2, axis=0)
    rhs_symbolic_tensor = keras.KerasTensor(shape=rhs.shape, dtype=dtype)
    rhs_tensor = keras.ops.convert_to_tensor(rhs, dtype=dtype)

    expected_sol = np.array([[1, 0], [-2, 0], [1, 1]])
    expected_sol = np.repeat(expected_sol[None, None], 2, axis=0)

    sol_symbolic_tensor = LinalgSolve()(matrix_symbolic_tensor, rhs_symbolic_tensor)
    assert tuple(sol_symbolic_tensor.shape) == tuple(expected_sol.shape)

    sol_tensor = LinalgSolve()(matrix_tensor, rhs_tensor)
    assert keras.backend.standardize_dtype(sol_tensor.dtype) == dtype
    assert_almost_equal(expected_sol, keras.ops.convert_to_numpy(sol_tensor), decimal=decimal)


def test_share_layer_all_weights_nok_original_layer_unbuilt():
    original_layer = Dense(3)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    with pytest.raises(ValueError):
        share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)


def test_share_layer_all_weights_nok_new_layer_built():
    original_layer = Dense(3)
    inp = Input((1,))
    original_layer(inp)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    new_layer(inp)
    with pytest.raises(ValueError):
        share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)


def test_share_layer_all_weights_ok():
    original_layer = Dense(3)
    inp = Input((1,))
    original_layer(inp)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)

    # check same weights
    assert len(original_layer.weights) == len(new_layer.weights)
    for w in original_layer.weights:
        new_w = [ww for ww in new_layer.weights if ww.name == w.name][0]
        assert w is new_w
