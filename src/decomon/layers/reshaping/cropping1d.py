from keras.layers import Cropping1D #type:ignore
from decomon.layers import  DecomonLinearLayer


class DecomonCropping1D(DecomonLinearLayer):
    layer: Cropping1D
    increasing = True
