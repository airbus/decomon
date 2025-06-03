from keras.layers import Cropping1D

from decomon.layers import DecomonLayer


class DecomonCropping1D(DecomonLayer):
    layer: Cropping1D
    linear = True
    increasing = True
    use_bias = False
