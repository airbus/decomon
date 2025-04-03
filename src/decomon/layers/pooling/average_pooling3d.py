from keras.layers import AveragePooling3D  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonAveragePooling3D(DecomonLinearLayer):
    layer: AveragePooling3D
    linear = True
    increasing = True
