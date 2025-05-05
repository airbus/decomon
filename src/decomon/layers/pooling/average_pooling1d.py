from keras.layers import AveragePooling1D

from decomon.layers import DecomonLinearLayer


class DecomonAveragePooling1D(DecomonLinearLayer):
    layer: AveragePooling1D
    linear = True
    increasing = True
