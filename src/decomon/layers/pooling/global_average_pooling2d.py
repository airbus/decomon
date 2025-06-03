from keras.layers import GlobalAveragePooling2D

from decomon.layers import DecomonLayer
from decomon.layers.utils import get_affine_representation_wo_bias
from decomon.types import Tensor


class DecomonGlobalAveragePooling2D(DecomonLayer):
    layer: GlobalAveragePooling2D
    linear = True
    increasing = True
    use_bias = False
