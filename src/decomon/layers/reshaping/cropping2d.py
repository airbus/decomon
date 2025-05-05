from keras.layers import Cropping2D

from decomon.layers import DecomonLinearLayer


class DecomonCropping2D(DecomonLinearLayer):
    layer: Cropping2D
    increasing = True
