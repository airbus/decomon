from keras.layers import LayerNormalization

from decomon.layers import DecomonLayer
from decomon.layers.utils import get_affine_representation_with_bias
from decomon.types import Tensor


class DecomonLayerNormalization(DecomonLayer):
    layer: LayerNormalization
    linear = True
    diagonal = True
