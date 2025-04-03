from keras.layers import Permute  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonPermute(DecomonLinearLayer):
    layer: Permute
    increasing = True
