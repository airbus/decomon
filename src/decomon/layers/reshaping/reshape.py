from keras.layers import Reshape

from decomon.layers import DecomonLinearLayer


class DecomonReshape(DecomonLinearLayer):
    layer: Reshape
    increasing = True
