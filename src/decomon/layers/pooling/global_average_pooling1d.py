from keras.layers import GlobalAveragePooling1D

from decomon.layers import DecomonLayer
from decomon.layers.utils import get_affine_representation_wo_bias
from decomon.types import Tensor


class DecomonGlobalAveragePooling1D(DecomonLayer):
    layer: GlobalAveragePooling1D
    linear = True
    increasing = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        return get_affine_representation_wo_bias(self.layer, diagonal=self.diagonal)
