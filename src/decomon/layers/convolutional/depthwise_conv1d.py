from keras.layers import DepthwiseConv1D

from .base_conv import DecomonBaseDepthwiseConv


class DecomonDepthwiseConv1D(DecomonBaseDepthwiseConv):
    layer: DepthwiseConv1D
