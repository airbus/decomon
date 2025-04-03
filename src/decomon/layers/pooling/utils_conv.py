from typing import Dict, Tuple

import keras  # type:ignore
import keras.ops as K  # type:ignore
import numpy as np  # type:ignore
from keras.layers import (  # type:ignore
    Conv2DTranspose,
    DepthwiseConv2D,
    Layer,
    MaxPooling2D,
)


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
    pooling: int
    kernel_pool: np.array.array

    pool_size_x, pool_size_y = config["pool_size"]
    pooling = pool_size_x * pool_size_y

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
    strides: Tuple[int] = config["strides"]
    padding: str = config["padding"]
    data_format: str = config["data_format"]

    layer_conv: DepthwiseConv2D = DepthwiseConv2D(
        depth_multiplier=filters,
        kernel_size=pool_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=data_format,
    )
    layer_conv.trainable = False
    layer_conv.kernel = kernel
    layer_conv.built = True
    # layer_conv.output.shape = (batch, in_channels*out_channel, w, h) if data_format=='channel_first
    # layer_conv.output.shape = (batch, w, h, in_channels*out_channel) if data_format=='channel_last

    return layer_conv, kernel


def get_backward_layer(layer: DepthwiseConv2D) -> Layer:
    dico_conv = layer.get_config()
    # dico_conv.pop("groups")
    # update filters to match input, pay attention to data_format
    if layer.data_format == "channels_first":  # better to use enum than raw str
        dico_conv["filters"] = 1  # input_shape[0]
    else:
        dico_conv["filters"] = 1  # input_shape[-1]

    dico_conv["use_bias"] = False

    # temporary fix

    # discard keys that start by depth
    # depth_keys = [e for e[:5]=='depth' for e in dico_conv.keys()]
    dico_conv.pop("depth_multiplier")
    dico_conv.pop("depthwise_initializer")
    dico_conv.pop("depthwise_regularizer")
    dico_conv.pop("depthwise_constraint")

    layer_backward = Conv2DTranspose.from_config(dico_conv)
    layer_backward.kernel = layer.kernel[:, :, :1, :]
    layer_backward.built = True

    return layer_backward


def get_maxpool_backward_hull(
    w_u_out_e,
    w_u_out_pos_e,
    w_u_out_neg_e,
    w_l_out_pos_e,
    w_l_out_neg_e,
    upper_max,
    lower_max,
    axis,
    get_affine_bounds_with_linear_block_inputs,
):
    # reshape lower_max and upper_max and update axis if necessary
    n_out = len(w_u_out_e.shape) - len(lower_max.shape)
    expand_shape = [-1] + list(lower_max.shape)[1:] + [1] * n_out
    lower_max_e = K.reshape(lower_max, expand_shape)  # same shape as w_u_out_e
    upper_max_e = K.reshape(upper_max, expand_shape)  # same shape as w_u_out_e

    if axis == -1:
        axis_ = len(lower_max.shape) - 1
    else:
        axis_ = axis

    lower_max_u_0 = lower_max_e * w_u_out_pos_e
    upper_max_u_0 = upper_max_e * w_u_out_pos_e
    _, _, w_u_0, b_u_0 = get_affine_bounds_with_linear_block_inputs(
        lower_max=lower_max_u_0, upper_max=upper_max_u_0, axis=axis_
    )

    lower_max_u_1 = -upper_max_e * w_u_out_neg_e
    upper_max_u_1 = -lower_max_e * w_u_out_neg_e
    w_l_1, b_l_1, _, _ = get_affine_bounds_with_linear_block_inputs(
        lower_max=lower_max_u_1, upper_max=upper_max_u_1, axis=axis_
    )

    w_u = w_u_0 - w_l_1
    b_u = b_u_0 - b_l_1

    #### lower bound
    lower_max_l_0 = lower_max_e * w_l_out_pos_e
    upper_max_l_0 = upper_max_e * w_l_out_pos_e
    w_l_0, b_l_0, _, _ = get_affine_bounds_with_linear_block_inputs(
        lower_max=lower_max_l_0, upper_max=upper_max_l_0, axis=axis_
    )
    lower_max_l_1 = -upper_max_e * w_l_out_neg_e
    upper_max_l_1 = -lower_max_e * w_l_out_neg_e
    _, _, w_u_1, b_u_1 = get_affine_bounds_with_linear_block_inputs(
        lower_max=lower_max_l_1, upper_max=upper_max_l_1, axis=axis_
    )

    w_l = w_l_0 - w_u_1
    b_l = b_l_0 - b_u_1

    return [w_l, b_l, w_u, b_u]  # add bias b_u_out, b_l_out
