from keras.layers import ZeroPadding1D  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonZeroPadding1D(DecomonLinearLayer):
    layer: ZeroPadding1D
    increasing = True
