from keras.layers import UpSampling3D

from decomon.layers import DecomonLayer


class DecomonUpSampling3D(DecomonLayer):
    layer: UpSampling3D
    linear = True
    increasing = True
