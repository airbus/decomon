from keras.layers import DepthwiseConv2D  # type:ignore

from .base_conv import DecomonBaseDepthwiseConv


class DecomonDepthwiseConv2D(DecomonBaseDepthwiseConv):
    layer: DepthwiseConv2D
