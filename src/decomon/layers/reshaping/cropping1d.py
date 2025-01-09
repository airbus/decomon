from keras.layers import Cropping1D
from decomon.layers import DecomonLayer, DecomonLinearLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonCropping1D(DecomonLinearLayer):
    layer: Cropping1D
    increasing = True
