from keras.layers import ZeroPadding1D

from decomon.layers import DecomonLinearLayer


class DecomonZeroPadding1D(DecomonLinearLayer):
    layer: ZeroPadding1D
    increasing = True
