from keras.layers import ZeroPadding1D
from decomon.layers import DecomonLayer, DecomonLinearLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonZeroPadding1D(DecomonLinearLayer):
    layer: ZeroPadding1D
    increasing = True
