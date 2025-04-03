from keras.layers import UpSampling3D  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonUpSampling3D(DecomonLinearLayer):
    layer: UpSampling3D
    linear = True
    increasing = True
