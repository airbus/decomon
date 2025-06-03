from keras.layers import UpSampling2D

from decomon.layers import DecomonLayer


class DecomonUpSampling2D(DecomonLayer):
    layer: UpSampling2D
    linear = True
    increasing = True
    use_bias = False
