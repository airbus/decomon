from keras.layers import Cropping2D

from decomon.layers import DecomonLayer


class DecomonCropping2D(DecomonLayer):
    layer: Cropping2D
    linear = True
    increasing = True
    use_bias = False
