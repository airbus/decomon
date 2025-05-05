from keras.layers import DepthwiseConv2D

from .base_conv import DecomonBaseDepthwiseConv


class DecomonDepthwiseConv2D(DecomonBaseDepthwiseConv):
    layer: DepthwiseConv2D
