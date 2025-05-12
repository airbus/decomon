from keras.layers import UpSampling1D

from decomon.layers import DecomonLayer


class DecomonUpSampling1D(DecomonLayer):
    layer: UpSampling1D
    linear = True
    increasing = True
