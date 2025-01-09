from keras.layers import Cropping2D
from decomon.layers import DecomonLayer, DecomonLinearLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonCropping2D(DecomonLinearLayer):
    layer: Cropping2D
    increasing = True
