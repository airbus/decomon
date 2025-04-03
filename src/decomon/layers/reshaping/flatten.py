from keras.layers import Flatten  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonFlatten(DecomonLinearLayer):
    layer: Flatten
    increasing = True
