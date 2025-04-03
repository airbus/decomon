from keras.layers import ZeroPadding2D  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonZeroPadding2D(DecomonLinearLayer):
    layer: ZeroPadding2D
    increasing = True
