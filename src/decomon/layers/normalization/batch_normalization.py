from keras.layers import BatchNormalization
from decomon.layers import DecomonLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_with_bias

class DecomonBatchNormalization(DecomonLayer):
    layer: BatchNormalization
    linear=True
    diagonal=True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:

        return get_affine_representation_with_bias(self.layer, diagonal = self.diagonal)
