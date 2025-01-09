from keras.layers import RepeatVector
from decomon.layers import DecomonLayer, DecomonLinearLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonRepeatVector(DecomonLinearLayer):
    layer: RepeatVector
    increasing = True

