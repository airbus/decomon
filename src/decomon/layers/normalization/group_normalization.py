from keras.layers import GroupNormalization

from decomon.layers import DecomonLayer
from decomon.layers.utils import get_affine_representation_with_bias
from decomon.types import Tensor


class DecomonGroupNormalization(DecomonLayer):
    layer: GroupNormalization
    linear = True
    diagonal = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        return get_affine_representation_with_bias(self.layer, diagonal=self.diagonal)
