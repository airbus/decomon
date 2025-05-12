from keras.layers import AveragePooling1D

from decomon.layers import DecomonLayer


class DecomonAveragePooling1D(DecomonLayer):
    layer: AveragePooling1D
    linear = True
    increasing = True
