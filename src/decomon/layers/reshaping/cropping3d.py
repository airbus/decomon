from keras.layers import Cropping3D #type:ignore
from decomon.layers import DecomonLinearLayer


class DecomonCropping3D(DecomonLinearLayer):
    layer: Cropping3D
    increasing = True
