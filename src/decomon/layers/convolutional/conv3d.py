from keras.layers import Conv3D

from .base_conv import DecomonBaseConv


class DecomonConv3D(DecomonBaseConv):
    layer: Conv3D
