from keras.layers import Flatten
from decomon.layers import DecomonLayer, DecomonLinearLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonFlatten(DecomonLinearLayer):
    layer: Flatten
    increasing = True
