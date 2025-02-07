from keras.layers import Cropping3D
from decomon.layers import DecomonLinearLayer


class DecomonCropping3D(DecomonLinearLayer):
    layer: Cropping3D
    increasing = True
