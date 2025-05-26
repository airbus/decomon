import pytest
import torch
from keras.layers import Dense


def _test_backward_dense(units, input_shape, method, wo_linearity, helpers):
    keras_layer = Dense(units)
    torch_layer = torch.nn.Linear(input_shape[-1], units)
    helpers.check_layer(keras_layer, torch_layer, input_shape, method=method, wo_linearity=wo_linearity, decimal=5)


def _test_backward_dense_empirical(units, input_shape, method, wo_linearity, helpers):
    keras_layer = Dense(units)
    helpers.empirical_check_layer(keras_layer, input_shape, method=method, wo_linearity=wo_linearity, decimal=5)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_dense(method, helpers):
    input_shape = (10,)
    units = 100
    _test_backward_dense(units, input_shape, method, wo_linearity=False, helpers=helpers)
    _test_backward_dense(units, input_shape, method, wo_linearity=True, helpers=helpers)

    input_shape = (11,)
    units = 101
    _test_backward_dense(units, input_shape, method, wo_linearity=False, helpers=helpers)
    _test_backward_dense(units, input_shape, method, wo_linearity=True, helpers=helpers)


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_dense_empirical(method, helpers):
    input_shape = (10,)
    units = 100
    _test_backward_dense_empirical(units, input_shape, method, wo_linearity=False, helpers=helpers)
    _test_backward_dense_empirical(units, input_shape, method, wo_linearity=True, helpers=helpers)

    input_shape = (11,)
    units = 101
    _test_backward_dense_empirical(units, input_shape, method, wo_linearity=False, helpers=helpers)
    _test_backward_dense_empirical(units, input_shape, method, wo_linearity=True, helpers=helpers)
