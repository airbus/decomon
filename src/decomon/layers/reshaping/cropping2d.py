from keras.layers import Cropping2D  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonCropping2D(DecomonLinearLayer):
    layer: Cropping2D
    increasing = True
