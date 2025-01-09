from keras.layers import Cropping3D
from decomon.layers import DecomonLayer, DecomonLinearLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonCropping3D(DecomonLinearLayer):
    layer: Cropping3D
    increasing = True
