import sys

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


def _test_backward_MaxPooling2D(pool_size, strides, padding, input_shape, data_format, method, helpers):
    keras_layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)
    helpers.empirical_check_layer(keras_layer, input_shape, method=method, decimal=4)


def _test_backward_AveragePooling2D_linear(pool_size, strides, padding, input_shape, method, helpers):
    keras_layer = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)
    helpers.check_layer_linear(keras_layer, input_shape, method, decimal=6)


def _test_backward_AveragePooling1D(pool_size, strides, padding, input_shape, method, helpers):
    keras_layer = AveragePooling1D(pool_size=pool_size, strides=strides, padding=padding)
    helpers.check_layer_linear(keras_layer, input_shape, method, decimal=6)


def _test_backward_AveragePooling3D(pool_size, strides, padding, input_shape, method, helpers):
    keras_layer = AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding)
    helpers.check_layer_linear(keras_layer, input_shape, method, decimal=6)


#####
def _test_backward_GlobalAveragePooling2D_linear(input_shape, method, helpers):
    keras_layer = GlobalAveragePooling2D()
    helpers.check_layer_linear(keras_layer, input_shape, method, decimal=6)


def _test_backward_GlobalAveragePooling1D(input_shape, method, helpers):
    keras_layer = GlobalAveragePooling1D()
    helpers.check_layer_linear(keras_layer, input_shape, method, decimal=6)


def _test_backward_GlobalAveragePooling3D(input_shape, method, helpers):
    keras_layer = GlobalAveragePooling3D()
    helpers.check_layer_linear(keras_layer, input_shape, method, decimal=6)


@pytest.mark.parametrize(
    "method",
    [
        "forward-affine",
        "forward-hybrid",
        "crown-forward-ibp",
        "crown",
    ],
)
@pytest.mark.parametrize(
    "data_format",
    ["channels_first", "channels_last"],
)
@pytest.mark.parametrize(
    "padding",
    ["valid", "same"],
)
def test_backward_MaxPooling2D(method, data_format, padding, helpers):
    if (
        padding == "same"
        and sys.platform == "darwin"
        and method
        in (
            "forward-affine",
            "forward-hybrid",
        )
    ):
        pytest.skip("Wrong forward affine bounds for maxpooling2d with padding == 'same' on macos github runners.")
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (1, 10, 10)
    else:
        input_shape = (10, 10, 1)
    pool_size = (2, 2)
    strides = 1
    _test_backward_MaxPooling2D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        input_shape=input_shape,
        method=method,
        helpers=helpers,
    )


@pytest.mark.parametrize("method", ["crown"])
def test_backward_AveragePooling2D_linear(method, helpers):
    keras.config.set_image_data_format("channels_first")
    input_shape = (1, 10, 10)
    pool_size = (2, 2)
    strides = 1
    padding = "valid"
    _test_backward_AveragePooling2D_linear(pool_size, strides, padding, input_shape, method, helpers)

    input_shape = (1, 10, 10)
    pool_size = (2, 2)
    strides = 1
    padding = "same"
    _test_backward_AveragePooling2D_linear(pool_size, strides, padding, input_shape, method, helpers)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_GlobalAveragePooling2D_linear(method, helpers):
    keras.config.set_image_data_format("channels_first")

    input_shape = (1, 10, 10)
    _test_backward_GlobalAveragePooling2D_linear(input_shape, method, helpers)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_AveragePooling1D_linear(method, helpers):
    keras.config.set_image_data_format("channels_first")
    input_shape = (1, 10)
    pool_size = (2,)
    strides = 1
    padding = "valid"
    _test_backward_AveragePooling1D(pool_size, strides, padding, input_shape, method, helpers)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_GlobalAveragePooling1D_linear(method, helpers):
    keras.config.set_image_data_format("channels_first")
    input_shape = (1, 10)
    _test_backward_GlobalAveragePooling1D(input_shape, method, helpers)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_AveragePooling3D_linear(method, helpers):
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
    _test_backward_AveragePooling3D(pool_size, strides, padding, input_shape, method, helpers)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_GlobalAveragePooling3D_linear(method, helpers):
    keras.config.set_image_data_format("channels_first")
    # skip tests on MPS device as Conv3DTranspose is not implemented
    if keras.config.backend() == "torch":
        import torch

        if torch.backends.mps.is_available():
            pytest.skip("skip tests on MPS device as AveragePooling3D is not implemented")
    input_shape = (1, 10, 11, 10)
    _test_backward_GlobalAveragePooling3D(input_shape, method, helpers)
