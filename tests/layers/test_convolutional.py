import pytest

import keras
from keras.layers import (
    Conv2D, Conv1D, Conv3D, DepthwiseConv2D, DepthwiseConv1D

)
import torch

from .conftest_layer import check_layer

def _test_backward_depthwise_conv2D(depth_multiplier, kernel_size, strides, padding, input_shape, method):

    keras_layer = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, depth_multiplier=1)
    if padding == 'same':
        padding_t = 1
    else:
        padding_t = 0
    torch_layer = torch.nn.Conv2d(input_shape[0], input_shape[0], kernel_size=kernel_size, stride=strides, padding=padding_t, groups=input_shape[0])
    check_layer(keras_layer, torch_layer, input_shape, method=method, axis_to_permute_kernel=(2, 3, 0, 1), decimal=5)

def _test_backward_depthwise_conv1D(depth_multiplier, kernel_size, strides, padding, input_shape, method):

    keras_layer = DepthwiseConv1D(kernel_size=kernel_size, strides=strides, padding=padding, depth_multiplier=1)
    if padding == 'same':
        padding_t = 1
    else:
        padding_t = 0
    torch_layer = torch.nn.Conv1d(input_shape[0], input_shape[0], kernel_size=kernel_size, stride=strides, padding=padding_t, groups=input_shape[0])
    check_layer(keras_layer, torch_layer, input_shape, method=method, axis_to_permute_kernel=(2, 0, 1), decimal=5)

def _test_backward_conv3d(filters, kernel_size, strides, padding, input_shape, method):

    # data_format == 'channels_first'
    keras_layer = Conv3D(filters,kernel_size,padding=padding, data_format='channels_first')
    if padding == 'same':
        padding_t = 1
    else:
        padding_t = 0
    torch_layer = torch.nn.Conv3d(input_shape[0], filters, kernel_size=kernel_size, stride=strides, padding=padding_t)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)

def _test_backward_conv2d(filters, kernel_size, strides, padding, input_shape, method):

    # data_format == 'channels_first'
    keras_layer = Conv2D(filters,kernel_size,padding=padding, data_format='channels_first')
    if padding == 'same':
        padding_t = 1
    else:
        padding_t = 0
    torch_layer = torch.nn.Conv2d(input_shape[0], filters, kernel_size=kernel_size, stride=strides, padding=padding_t)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)

def _test_backward_conv1d(filters, kernel_size, strides, padding, input_shape, method):

    # data_format == 'channels_first'
    keras_layer = Conv1D(filters,kernel_size,padding=padding, data_format='channels_first')
    if padding == 'same':
        padding_t = 1
    else:
        padding_t = 0
    torch_layer = torch.nn.Conv1d(input_shape[0], filters, kernel_size=kernel_size, stride=strides, padding=padding_t)
    check_layer(keras_layer, torch_layer, input_shape, method=method, axis_to_permute_kernel=(2, 1, 0), decimal=5)


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

@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_Conv1D(method):
    input_shape = (2, 10)
    filters = 32
    kernel_size = 3
    strides = 1
    padding='same'
    _test_backward_conv1d(filters, kernel_size, strides, padding, input_shape, method)

    padding='valid'
    _test_backward_conv1d(filters, kernel_size, strides, padding, input_shape, method)

@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_Conv3D(method):

    # skip tests on MPS device as Conv3DTranspose is not implemented
    if keras.config.backend() == "torch":
        import torch

        if torch.backends.mps.is_available():
            pytest.skip("skip tests on MPS device as Conv3DTranspose is not implemented")

    input_shape = (2, 10, 10, 11)
    filters = 32
    kernel_size = (3,3, 2)
    strides = 1
    padding='same'
    _test_backward_conv3d(filters, kernel_size, strides, padding, input_shape, method)

@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_DepthwiseConv2D(method):
    input_shape = (30, 10, 10)
    depth_multiplier = 1
    kernel_size = (3,3)
    strides = 1
    padding='same'
    _test_backward_depthwise_conv2D(depth_multiplier, kernel_size, strides, padding, input_shape, method)

    input_shape = (30, 10, 10)
    depth_multiplier = 2
    kernel_size = (3,3)
    strides = 1
    padding='valid'
    _test_backward_depthwise_conv2D(depth_multiplier, kernel_size, strides, padding, input_shape, method)

@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_DepthwiseConv1D(method):
    input_shape = (2, 10)
    kernel_size = 3
    strides = 1
    depth_multiplier = 1
    padding='same'
    _test_backward_depthwise_conv1D(depth_multiplier, kernel_size, strides, padding, input_shape, method)

    input_shape = (2, 10)
    kernel_size = 3
    strides = 1
    depth_multiplier = 1
    strides = 1
    padding='valid'
    _test_backward_depthwise_conv1D(depth_multiplier, kernel_size, strides, padding, input_shape, method)


