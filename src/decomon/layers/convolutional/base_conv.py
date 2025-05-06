from typing import Any

import keras.ops as K
from keras.src.layers.convolutional.base_conv import BaseConv
from keras.src.layers.convolutional.base_depthwise_conv import BaseDepthwiseConv

from decomon.layers.layer import DecomonLinearLayer

from .utils import ConvKernelConstraint, DepthwiseConvKernelConstraint


class DecomonBaseConv(DecomonLinearLayer):
    def __init__(
        self,
        layer: BaseConv,
        *args: Any,
        **kwargs: Any,
    ):
        layer_pos = ConvKernelConstraint(layer=layer, ops=K.maximum, add_bias=True)
        layer_neg = ConvKernelConstraint(layer=layer, ops=K.minimum, add_bias=False)
        super().__init__(layer=layer, layer_pos=layer_pos, layer_neg=layer_neg, *args, **kwargs)  # type: ignore


class DecomonBaseDepthwiseConv(DecomonLinearLayer):
    def __init__(
        self,
        layer: BaseDepthwiseConv,
        *args: Any,
        **kwargs: Any,
    ):
        layer_pos = DepthwiseConvKernelConstraint(layer=layer, ops=K.maximum, add_bias=True)
        layer_neg = DepthwiseConvKernelConstraint(layer=layer, ops=K.minimum, add_bias=False)
        super().__init__(layer=layer, layer_pos=layer_pos, layer_neg=layer_neg, *args, **kwargs)  # type: ignore
