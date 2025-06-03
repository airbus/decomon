from keras.layers import ZeroPadding1D

from decomon.layers import DecomonLayer


class DecomonZeroPadding1D(DecomonLayer):
    layer: ZeroPadding1D
    linear = True
    increasing = True
    use_bias = False
