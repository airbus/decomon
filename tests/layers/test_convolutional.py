import pytest

from keras.layers import (
    Conv2D

)
import torch

from .conftest_layer import check_layer

def _test_backward_conv2d(filters, kernel_size, strides, padding, input_shape, method):

    # data_format == 'channels_first'
    keras_layer = Conv2D(filters,kernel_size,padding=padding, data_format='channels_first')
    if padding == 'same':
        padding_t = 1
    else:
        padding_t = 0
    torch_layer = torch.nn.Conv2d(input_shape[0], filters, kernel_size=kernel_size, stride=strides, padding=padding_t)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_Conv2D(method):
    input_shape = (2, 10, 10)
    filters = 32
    kernel_size = (3,3)
    strides = 1
    padding='same'
    _test_backward_conv2d(filters, kernel_size, strides, padding, input_shape, method)

    input_shape = (2, 10, 10)
    filters = 32
    kernel_size = (3,3)
    strides = 1
    padding='valid'
    _test_backward_conv2d(filters, kernel_size, strides, padding, input_shape, method)

