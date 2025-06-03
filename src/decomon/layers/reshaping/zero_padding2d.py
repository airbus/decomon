from keras.layers import ZeroPadding2D

from decomon.layers import DecomonLayer


class DecomonZeroPadding2D(DecomonLayer):
    layer: ZeroPadding2D
    linear = True
    increasing = True
    use_bias = False
