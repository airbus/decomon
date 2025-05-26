import keras
import numpy as np
import pytest
import torch

# linear pooling
from keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    MaxPooling2D,
)

from .conftest import check_layer, check_layer_linear, empirical_check_layer


def _test_backward_MaxPooling2D(pool_size, strides, padding, input_shape, method):
    keras_layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=4)


def _test_backward_AveragePooling2D_linear(pool_size, strides, padding, input_shape, method):
    keras_layer = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)
    check_layer_linear(keras_layer, input_shape, method, decimal=6)


def _test_backward_AveragePooling1D(pool_size, strides, padding, input_shape, method):
    keras_layer = AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding)
    check_layer_linear(keras_layer, input_shape, method, decimal=6)


def _test_backward_AveragePooling3D(pool_size, strides, padding, input_shape, method):
    keras_layer = AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding)
    check_layer_linear(keras_layer, input_shape, method, decimal=6)


#####
def _test_backward_GlobalAveragePooling2D_linear(input_shape, method):
    keras_layer = GlobalAveragePooling2D()
    check_layer_linear(keras_layer, input_shape, method, decimal=6)


def _test_backward_GlobalAveragePooling1D(input_shape, method):
    keras_layer = GlobalAveragePooling1D()
    check_layer_linear(keras_layer, input_shape, method, decimal=6)


def _test_backward_GlobalAveragePooling3D(input_shape, method):
    keras_layer = GlobalAveragePooling3D()
    check_layer_linear(keras_layer, input_shape, method, decimal=6)


@pytest.mark.parametrize(
    "method, data_format",
    [
        ("forward-affine", "channels_first"),
        ("forward-hybrid", "channels_first"),
        ("crown-forward-ibp", "channels_first"),
        ("crown", "channels_first"),
        ("forward-affine", "channels_last"),
        ("forward-hybrid", "channels_last"),
        ("crown-forward-ibp", "channels_last"),
        ("crown", "channels_last"),
    ],
)
def test_backward_MaxPooling2D(method, data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (1, 10, 10)
    else:
        input_shape = (10, 10, 1)
    pool_size = (2, 2)
    strides = 1
    padding = "valid"
    _test_backward_MaxPooling2D(pool_size, strides, padding, input_shape, method)
    if data_format == "channels_first":
        input_shape = (1, 10, 10)
    else:
        input_shape = (10, 10, 1)
    pool_size = (2, 2)
    strides = 1
    padding = "same"
    _test_backward_MaxPooling2D(pool_size, strides, padding, input_shape, method)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_AveragePooling2D_linear(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (1, 10, 10)
    pool_size = (2, 2)
    strides = 1
    padding = "valid"
    _test_backward_AveragePooling2D_linear(pool_size, strides, padding, input_shape, method)

    input_shape = (1, 10, 10)
    pool_size = (2, 2)
    strides = 1
    padding = "same"
    _test_backward_AveragePooling2D_linear(pool_size, strides, padding, input_shape, method)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_GlobalAveragePooling2D_linear(method):
    keras.config.set_image_data_format("channels_first")

    input_shape = (1, 10, 10)
    _test_backward_GlobalAveragePooling2D_linear(input_shape, method)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_AveragePooling1D_linear(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (1, 10)
    pool_size = (2,)
    strides = 1
    padding = "valid"
    _test_backward_AveragePooling1D(pool_size, strides, padding, input_shape, method)

    input_shape = (1, 10)
    pool_size = (2,)
    strides = 1
    padding = "same"
    _test_backward_AveragePooling1D(pool_size, strides, padding, input_shape, method)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_GlobalAveragePooling1D_linear(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (1, 10)
    _test_backward_GlobalAveragePooling1D(input_shape, method)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_AveragePooling3D_linear(method):
    keras.config.set_image_data_format("channels_first")
    # skip tests on MPS device as Conv3DTranspose is not implemented
    if keras.config.backend() == "torch":
        import torch

        if torch.backends.mps.is_available():
            pytest.skip("skip tests on MPS device as AveragePooling3D is not implemented")
    input_shape = (1, 10, 11, 10)
    pool_size = (2, 2, 2)
    strides = 1
    padding = "valid"
    _test_backward_AveragePooling3D(pool_size, strides, padding, input_shape, method)

    input_shape = (1, 10, 11, 10)
    pool_size = (2, 2, 2)
    strides = 1
    padding = "same"
    _test_backward_AveragePooling3D(pool_size, strides, padding, input_shape, method)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_GlobalAveragePooling3D_linear(method):
    keras.config.set_image_data_format("channels_first")
    # skip tests on MPS device as Conv3DTranspose is not implemented
    if keras.config.backend() == "torch":
        import torch

        if torch.backends.mps.is_available():
            pytest.skip("skip tests on MPS device as AveragePooling3D is not implemented")
    input_shape = (1, 10, 11, 10)
    _test_backward_GlobalAveragePooling3D(input_shape, method)
