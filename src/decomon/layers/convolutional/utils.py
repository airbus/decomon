import warnings

import keras
import keras.ops as K
import numpy as np
from keras.layers import Conv2D, Input
from keras.ops.image import extract_patches

from decomon.types import Tensor


def get_toeplitz(conv_layer: Conv2D) -> Tensor:
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
