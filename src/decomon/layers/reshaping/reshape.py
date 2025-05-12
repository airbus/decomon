from keras.layers import Reshape

from decomon.layers import DecomonLayer


class DecomonReshape(DecomonLayer):
    layer: Reshape
    linear = True
    increasing = True
