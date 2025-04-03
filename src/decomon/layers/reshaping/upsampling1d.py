from keras.layers import UpSampling1D  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonUpSampling1D(DecomonLinearLayer):
    layer: UpSampling1D
    linear = True
    increasing = True
