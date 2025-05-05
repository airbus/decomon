from keras.layers import LayerNormalization

from decomon.layers import DecomonLayer
from decomon.layers.utils import get_affine_representation_with_bias
from decomon.types import Tensor


class DecomonLayerNormalization(DecomonLayer):
    layer: LayerNormalization
    linear = True
    diagonal = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        return get_affine_representation_with_bias(self.layer, diagonal=self.diagonal)
