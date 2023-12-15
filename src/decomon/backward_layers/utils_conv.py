import warnings

import keras
import keras.ops as K
import numpy as np
from keras.layers import Conv2D, Input
from keras.ops.image import extract_patches

from decomon.types import BackendTensor


def get_toeplitz(conv_layer: Conv2D, flatten: bool = True) -> BackendTensor:
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
    if conv_layer.dtype == "float16":
        warnings.warn("Loss of precision for float16 !")
    if conv_layer.data_format == "channels_last":
        return get_toeplitz_channels_last(conv_layer, flatten)
    else:
        return get_toeplitz_channels_first(conv_layer, flatten)


def get_toeplitz_channels_last(conv_layer: Conv2D, flatten: bool = True) -> BackendTensor:
    """Express formally the affine component of the convolution for data_format=channels_last
    Conv is a linear operator but its affine component is implicit
    we use im2col and extract_patches to express the affine matrix
    Note that this matrix is Toeplitz

    Args:
        conv_layer: Keras Conv2D layer or Decomon Conv2D layer
        flatten (optional): convert the affine component as a 2D matrix (n_in, n_out). Defaults to True.

    Returns:
         the affine operator W: conv(x)= Wx + bias
    """

    input = conv_layer.input
    if isinstance(input, keras.KerasTensor):
        input_shape = input.shape
    else:  # list of inputs
        input_shape = input[-1].shape
    _, w_in, h_in, c_in = input_shape
    output = conv_layer.output
    if isinstance(output, keras.KerasTensor):
        output_shape = output.shape
    else:  # list of outputs
        output_shape = output[-1].shape
    _, w_out, h_out, c_out = output_shape

    kernel_filter = conv_layer.kernel
    filter_size = kernel_filter.shape[0]

    diag = K.reshape(K.identity(w_in * h_in * c_in), (w_in * h_in * c_in, w_in, h_in, c_in))

    diag_patches = extract_patches(
        diag,
        size=[filter_size, filter_size],
        strides=list(conv_layer.strides),
        dilation_rate=list(conv_layer.dilation_rate),
        padding=conv_layer.padding,
    )

    diag_patches_ = K.reshape(diag_patches, (w_in, h_in, c_in, w_out, h_out, filter_size**2, c_in))

    shape = list(range(len(diag_patches_.shape)))
    shape[-1] -= 1
    shape[-2] += 1
    diag_patches_ = K.transpose(diag_patches_, shape)
    kernel = conv_layer.kernel  # (filter_size, filter_size, c_in, c_out)

    kernel = K.transpose(K.reshape(kernel, (filter_size**2, c_in, c_out)), (1, 2, 0))[
        None, None, None, None, None
    ]  # (1,1,c_in, c_out, filter_size^2)

    # element-wise multiplication is only compatible with float32
    diag_patches_ = K.cast(diag_patches_, "float32")
    kernel = K.cast(kernel, "float32")

    w = K.sum(K.expand_dims(diag_patches_, -2) * kernel, (5, 7))
    # kernel shape: (1,1,1,1,1,c_in, c_out, filter_size^2)
    # diag_patches shape: (w_in, h_in, c_in, w_out, h_out, c_in, 1,     filter_size^2)
    # (w_in, h_in, c_in, w_out, h_out, c_in, c_out, filter^2)

    w = K.cast(w, conv_layer.dtype)
    # cast w for dtype

    if flatten:
        return K.reshape(w, (w_in * h_in * c_in, w_out * h_out * c_out))
    else:
        return w


def get_toeplitz_channels_first(conv_layer: Conv2D, flatten: bool = True) -> BackendTensor:
    """Express formally the affine component of the convolution for data_format=channels_first
    Conv is a linear operator but its affine component is implicit
    we use im2col and extract_patches to express the affine matrix
    Note that this matrix is Toeplitz

    Args:
        conv_layer: Keras Conv2D layer or Decomon Conv2D layer
        flatten (optional): convert the affine component as a 2D matrix (n_in, n_out). Defaults to True.

    Returns:
        the affine operator W: conv(x)= Wx + bias
    """

    input = conv_layer.input
    if isinstance(input, keras.KerasTensor):
        input_shape = input.shape
    else:  # list of inputs
        input_shape = input[-1].shape
    _, c_in, w_in, h_in = input_shape
    output = conv_layer.output
    if isinstance(output, keras.KerasTensor):
        output_shape = output.shape
    else:  # list of outputs
        output_shape = output[-1].shape
    _, c_out, w_out, h_out = output_shape
    kernel_filter = conv_layer.kernel
    filter_size = kernel_filter.shape[0]

    diag = K.reshape(K.identity(w_in * h_in * c_in), (w_in * h_in * c_in, w_in, h_in, c_in))

    diag_patches = extract_patches(diag, [filter_size, filter_size], [1, 1], [1, 1], padding=conv_layer.padding)

    diag_patches_ = K.reshape(diag_patches, (w_in, h_in, c_in, w_out, h_out, filter_size**2, c_in))

    shape = list(range(len(diag_patches_.shape)))
    shape[-1] -= 1
    shape[-2] += 1
    diag_patches_ = K.transpose(diag_patches_, shape)
    kernel = conv_layer.kernel  # (filter_size, filter_size, c_in, c_out)

    kernel = K.transpose(K.reshape(kernel, (filter_size**2, c_in, c_out)), (1, 2, 0))[
        None, None, None, None, None
    ]  # (1,1,c_in, c_out, filter_size^2)

    w = K.sum(K.expand_dims(diag_patches_, -2) * kernel, (5, 7))
    # kernel shape: (1,      1,     1,   1,    1,    c_in, c_out, filter_size^2)
    # diag_patches shape: (w_in, h_in, c_in, w_out, h_out, c_in, 1,     filter_size^2)
    # (w_in, h_in, c_in, w_out, h_out, c_in, c_out, filter^2)

    w = K.cast(w, conv_layer.dtype)
    # cast w for dtype

    if flatten:
        return K.reshape(w, (w_in * h_in * c_in, w_out * h_out * c_out))
    else:
        return w
