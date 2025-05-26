from typing import Any

import keras.ops as K
from keras.src.layers.convolutional.base_conv import BaseConv
from keras.src.layers.convolutional.base_depthwise_conv import BaseDepthwiseConv

from decomon.layers.layer import DecomonLayer

from .utils import ConvKernelConstraint, DepthwiseConvKernelConstraint


class DecomonBaseConv(DecomonLayer):
    linear = True

    def __init__(
        self,
        layer: BaseConv,
        *args: Any,
        **kwargs: Any,
    ):
        layer_pos = ConvKernelConstraint(layer=layer, ops=K.maximum)
        layer_neg = ConvKernelConstraint(layer=layer, ops=K.minimum)
        super().__init__(layer=layer, layer_pos=layer_pos, layer_neg=layer_neg, *args, **kwargs)  # type: ignore


class DecomonBaseDepthwiseConv(DecomonLayer):
    linear = True

    def __init__(
        self,
        layer: BaseDepthwiseConv,
        *args: Any,
        **kwargs: Any,
    ):
        layer_pos = DepthwiseConvKernelConstraint(layer=layer, ops=K.maximum)
        layer_neg = DepthwiseConvKernelConstraint(layer=layer, ops=K.minimum)
        super().__init__(layer=layer, layer_pos=layer_pos, layer_neg=layer_neg, *args, **kwargs)  # type: ignore
