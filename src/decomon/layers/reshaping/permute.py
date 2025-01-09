from keras.layers import Permute
from decomon.layers import DecomonLayer, DecomonLinearLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonPermute(DecomonLinearLayer):
    layer: Permute
    increasing = True

