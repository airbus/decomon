from keras.layers import AveragePooling2D

from decomon.layers import DecomonLayer


class DecomonAveragePooling2D(DecomonLayer):
    layer: AveragePooling2D
    linear = True
    increasing = True
    use_bias = False
