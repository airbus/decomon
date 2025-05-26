import keras
import pytest
import torch
from keras.layers import Dense

from .conftest import check_layer, empirical_check_layer


def _test_backward_dense(units, input_shape, method, wo_linearity):
    keras_layer = Dense(units)
    torch_layer = torch.nn.Linear(input_shape[-1], units)
    check_layer(keras_layer, torch_layer, input_shape, method=method, wo_linearity=wo_linearity, decimal=5)


def _test_backward_dense_empirical(units, input_shape, method, wo_linearity):
    keras_layer = Dense(units)
    empirical_check_layer(keras_layer, input_shape, method=method, wo_linearity=wo_linearity, decimal=5)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_dense(method):
    input_shape = (10,)
    units = 100
    _test_backward_dense(units, input_shape, method, wo_linearity=False)
    _test_backward_dense(units, input_shape, method, wo_linearity=True)

    input_shape = (11,)
    units = 101
    _test_backward_dense(units, input_shape, method, wo_linearity=False)
    _test_backward_dense(units, input_shape, method, wo_linearity=True)


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_dense_empirical(method):
    input_shape = (10,)
    units = 100
    _test_backward_dense_empirical(units, input_shape, method, wo_linearity=False)
    _test_backward_dense_empirical(units, input_shape, method, wo_linearity=True)

    input_shape = (11,)
    units = 101
    _test_backward_dense_empirical(units, input_shape, method, wo_linearity=False)
    _test_backward_dense_empirical(units, input_shape, method, wo_linearity=True)
