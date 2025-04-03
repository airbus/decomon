from keras.layers import AveragePooling2D  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonAveragePooling2D(DecomonLinearLayer):
    layer: AveragePooling2D
    increasing = True
