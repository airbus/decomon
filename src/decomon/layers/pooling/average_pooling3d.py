from keras.layers import AveragePooling3D

from decomon.layers import DecomonLayer


class DecomonAveragePooling3D(DecomonLayer):
    layer: AveragePooling3D
    linear = True
    increasing = True
    use_bias = False
