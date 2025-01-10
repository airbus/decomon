import warnings

import keras
import keras.ops as K
import numpy as np
from keras.layers import Conv2D, Input
from keras.ops.image import extract_patches
from keras.layers import Wrapper
from keras.src.layers.convolutional.base_conv import BaseConv

from typing import Optional, Any
from decomon.types import Tensor

class Conv_kernel_constraint(Wrapper):

    def __init__(self, layer:BaseConv, ops=K.maximum, add_bias=True, **kwargs:Any):
        super().__init__(layer=layer, **kwargs)
        self.ops = ops
        self.add_bias = add_bias
    
    def call(self, inputs: list[Tensor]) -> list[Tensor]:

        y:Tensor =  K.conv(
            inputs,
            kernel = self.ops(0, self.layer.kernel),
            strides=list(self.layer.strides),
            padding=self.layer.padding,
            dilation_rate=self.layer.dilation_rate,
            data_format=self.layer.data_format,
        ) 

        if self.add_bias:
            y += self.layer(0*inputs)

        return y


def get_toeplitz_from_layer(conv_layer: Conv2D) -> Tensor:

    kernel = conv_layer.kernel
    input_shape = list(conv_layer.input.shape[1:])
    output_shape = list(conv_layer.output.shape[1:])
    config = conv_layer.get_config()

    return get_toeplitz(kernel, input_shape, output_shape, config)

    """Express formally the affine component of the convolution
    Conv is a linear operator but its affine component is implicit
    we use im2col and extract_patches to express the affine matrix
    Note that this matrix is Toeplitz

    Args:
        conv_layer: Keras Conv2D layer or Decomon Conv2D layer
        flatten (optional): convert the affine component as a 2D matrix (n_in, n_out). Defaults to True.

    Returns:
         the affine operator W: conv(x)= Wx + bias
    """

    input_shape = list(conv_layer.input.shape[1:])
    output_shape = list(conv_layer.output.shape[1:])
    N = np.prod(input_shape)

    diag = K.reshape(K.eye(N), [-1] + input_shape)

    size = conv_layer.kernel_size
    strides = conv_layer.strides
    dilation_rate = conv_layer.dilation_rate
    padding = conv_layer.padding
    data_format = conv_layer.data_format

    diag_patches = extract_patches(
        images=diag, size=size, strides=strides, dilation_rate=dilation_rate, padding=padding, data_format=data_format
    )

    if data_format == "channels_first":
        output_shape_ = output_shape[1:]
    else:
        output_shape_ = output_shape[:-1]

    toeplitz = K.reshape(diag_patches, [N] + list(conv_layer.kernel.shape)[:-1] + [1] + output_shape_)
    kernel_ext = K.reshape(conv_layer.kernel, [1] + list(conv_layer.kernel.shape) + [1] * len(output_shape_))
    axis_to_sum = [i + 1 for i in range(len(conv_layer.kernel.shape) - 1)]

    result = K.sum(toeplitz * kernel_ext, axis_to_sum)  # (N_in, n_filter, output_shape_)

    if data_format == "channels_last":
        axis_permute = (0, 2, 3, 1)
        result = K.transpose(result, axis_permute)

    result = K.reshape(result, input_shape + output_shape)

    return result


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
