from keras.layers import Permute

from decomon.layers import DecomonLinearLayer


class DecomonPermute(DecomonLinearLayer):
    layer: Permute
    increasing = True
