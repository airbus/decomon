from keras.layers import GlobalAveragePooling3D

from decomon.layers import DecomonLayer
from decomon.layers.utils import get_affine_representation_wo_bias
from decomon.types import Tensor


class DecomonGlobalAveragePooling3D(DecomonLayer):
    layer: GlobalAveragePooling3D
    linear = True
    increasing = True
    use_bias = False
