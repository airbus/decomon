from keras.layers import Flatten
from decomon.layers import DecomonLinearLayer


class DecomonFlatten(DecomonLinearLayer):
    layer: Flatten
    increasing = True
