from keras.layers import ZeroPadding3D
from decomon.layers import DecomonLayer, DecomonLinearLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonZeroPadding3D(DecomonLinearLayer):
    layer: ZeroPadding3D
    increasing = True
