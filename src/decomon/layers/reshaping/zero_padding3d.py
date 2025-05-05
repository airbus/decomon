from keras.layers import ZeroPadding3D

from decomon.layers import DecomonLinearLayer


class DecomonZeroPadding3D(DecomonLinearLayer):
    layer: ZeroPadding3D
    increasing = True
