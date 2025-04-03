from typing import Any

import keras.ops as K  # type:ignore
import numpy as np  # type:ignore
from keras.layers import Conv2D  # type:ignore
from keras.layers import Wrapper  # type:ignore
from keras.ops.image import extract_patches  # type:ignore
from keras.src.layers.convolutional.base_conv import BaseConv  # type:ignore
from keras.src.layers.convolutional.base_depthwise_conv import (
    BaseDepthwiseConv,  # type:ignore
)

from decomon.types import Tensor


class Conv_kernel_constraint(Wrapper):
    def __init__(self, layer: BaseConv, ops=K.maximum, add_bias=True, **kwargs: Any):
        super().__init__(layer=layer, **kwargs)
        self.ops = ops
        self.add_bias = add_bias

        # list every attributes of conv
        self.kernel_ = self.layer.kernel
        self.use_bias = self.layer.use_bias and self.add_bias
        if self.use_bias:
            self.bias_ = self.layer.bias
        self.strides = self.layer.strides
        self.padding = self.layer.padding
        self.dilation_rate = self.layer.dilation_rate
        self.data_format = self.layer.data_format

    @property
    def kernel(self):
        return self.ops(self.kernel_, 0)

    @property
    def bias(self):
        if self.layer.use_bias and self.add_bias:
            return self.bias_
        return None

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        y: Tensor = K.conv(
            inputs,
            kernel=self.kernel,
            strides=list(self.strides),
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.add_bias:
            y += self.layer(0 * inputs)

        return y

    def get_config(self):
        return self.layer.get_config()


class DepthwiseConv_kernel_constraint(Wrapper):
    def __init__(self, layer: BaseDepthwiseConv, ops=K.maximum, add_bias=True, **kwargs: Any):
        super().__init__(layer=layer, **kwargs)
        self.ops = ops
        self.add_bias = add_bias

        # list every attributes of conv
        self.kernel_ = self.layer.kernel
        self.use_bias = self.layer.use_bias and self.add_bias
        if self.use_bias:
            self.bias_ = self.layer.bias
        self.strides = self.layer.strides
        self.padding = self.layer.padding
        self.dilation_rate = self.layer.dilation_rate
        self.data_format = self.layer.data_format

    @property
    def kernel(self):
        return self.ops(self.kernel_, 0)

    @property
    def bias(self):
        if self.layer.use_bias and self.add_bias:
            return self.bias_
        return None

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def _get_input_channel(self, input_shape):
        return self.layer._get_input_channel(input_shape)

    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        y = K.depthwise_conv(
            inputs,
            kernel=self.kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format,
        )

        if self.add_bias:
            y += self.layer(0 * inputs)

        return y

    def get_config(self):
        return self.layer.get_config()


def get_toeplitz_from_layer(conv_layer: Conv2D) -> Tensor:
    kernel = conv_layer.kernel
    input_shape = list(conv_layer.input.shape[1:])
    output_shape = list(conv_layer.output.shape[1:])
    config = conv_layer.get_config()

    return get_toeplitz(kernel, input_shape, output_shape, config)


def get_toeplitz(kernel, input_shape, output_shape, config) -> Tensor:
    """Express formally the affine component of the convolution
    Conv is a linear operator but its affine component is implicit
    we use im2col and extract_patches to express the affine matrix
    Note that this matrix is Toeplitz

    Args:
        kernel:
        input_shape:
        output_shape:
        config

    Returns:
         the affine operator W: conv(x)= Wx + bias
    """

    # input_shape = list(conv_layer.input.shape[1:])
    # output_shape = list(conv_layer.output.shape[1:])
    N = np.prod(input_shape)

    diag = K.reshape(K.eye(N), [-1] + input_shape)

    size = config["kernel_size"]
    strides = config["strides"]
    dilation_rate = config["dilation_rate"]
    padding = config["padding"]
    data_format = config["data_format"]

    diag_patches = extract_patches(
        images=diag, size=size, strides=strides, dilation_rate=dilation_rate, padding=padding, data_format=data_format
    )

    if data_format == "channels_first":
        output_shape_ = output_shape[1:]
    else:
        output_shape_ = output_shape[:-1]

    toeplitz = K.reshape(diag_patches, [N] + list(kernel.shape)[:-1] + [1] + output_shape_)
    kernel_ext = K.reshape(kernel, [1] + list(kernel.shape) + [1] * len(output_shape_))
    axis_to_sum = [i + 1 for i in range(len(kernel.shape) - 1)]

    result = K.sum(toeplitz * kernel_ext, axis_to_sum)  # (N_in, n_filter, output_shape_)

    if data_format == "channels_last":
        axis_permute = (0, 2, 3, 1)
        result = K.transpose(result, axis_permute)

    result = K.reshape(result, input_shape + output_shape)

    return result
