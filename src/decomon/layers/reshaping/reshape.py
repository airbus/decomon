import keras.ops as K  # type:ignore
from keras.layers import Reshape  # type:ignore

from decomon.layers import DecomonLinearLayer


class DecomonReshape(DecomonLinearLayer):
    layer: Reshape
    increasing = True
