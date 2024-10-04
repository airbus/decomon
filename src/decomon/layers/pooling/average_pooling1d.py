from keras.layers import AveragePooling1D
from decomon.layers import DecomonLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias

class DecomonAveragePooling1D(DecomonLayer):
    layer: AveragePooling1D
    linear=True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:

        return get_affine_representation_wo_bias(self.layer, diagonal = self.diagonal)
