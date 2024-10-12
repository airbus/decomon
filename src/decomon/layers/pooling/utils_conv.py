import keras
from keras.layers import MaxPooling2D, DepthwiseConv2D

import numpy as np

from typing import Dict


def get_in_channels(layer) -> int:
    in_channels: int
    if layer.data_format == "channels_last":
        in_channels = layer.input.shape[-1]
    else:
        in_channels = layer.input.shape[1]

    return in_channels


def get_conv_op_config(config: Dict, in_channels: int) -> keras.Variable:

    pool_size_x: int
    pool_size_y: int
    padding: str
    pooling: int
    kernel_pool: array.array

    pool_size_x, pool_size_y = config["pool_size"]
    padding = config["padding"]
    pooling = pool_size_x * pool_size_y

    # if padding=='same':
    #    raise NotImplementedError()
    # create the convolution layer to extract the Toeplitz matrix
    kernel_pool = np.repeat(
        np.transpose(np.eye(pooling).reshape((pooling, pool_size_x, pool_size_y)), (1, 2, 0))[:, :, None, :], 1, -2
    )
    # repeat along in_channels axis for compatibility with DepthwiseConv2D
    kernel_pool = np.repeat(kernel_pool, in_channels, 2)

    # warning data_format (channel_last or channel_first)

    return keras.Variable(kernel_pool, trainable=False)


def get_conv_op(layer: MaxPooling2D) -> DepthwiseConv2D:

    config: Dict = layer.get_config()
    in_channels = get_in_channels(layer)
    kernel: keras.Variable = get_conv_op_config(config, in_channels)

    # define convolution
    filters: int = np.prod(config["pool_size"])
    pool_size: tuple[int] = config["pool_size"]
    strides: tule[int] = config["strides"]
    padding: str = config["padding"]
    data_format: str = config["data_format"]

    layer_conv: DepthwiseConv2D = DepthwiseConv2D(
        depth_multiplier=filters, kernel_size=pool_size, strides=strides, padding=padding, use_bias=False
    )
    layer_conv.trainable = False
    layer_conv.kernel = kernel
    layer_conv.built = True

    return layer_conv, kernel
