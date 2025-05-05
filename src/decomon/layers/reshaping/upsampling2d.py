from keras.layers import UpSampling2D

from decomon.layers import DecomonLinearLayer


class DecomonUpSampling2D(DecomonLinearLayer):
    layer: UpSampling2D
    linear = True
    increasing = True
