from keras.layers import UpSampling2D  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonUpSampling2D(DecomonLinearLayer):
    layer: UpSampling2D
    linear = True
    increasing = True
