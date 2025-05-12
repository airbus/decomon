from keras.layers import Cropping3D

from decomon.layers import DecomonLayer


class DecomonCropping3D(DecomonLayer):
    layer: Cropping3D
    linear = True
    increasing = True
