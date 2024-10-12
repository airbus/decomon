from keras.layers import GlobalAveragePooling3D
from decomon.layers import DecomonLayer

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonGlobalAveragePooling3D(DecomonLayer):
    layer: GlobalAveragePooling3D
    linear = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:

        return get_affine_representation_wo_bias(self.layer, diagonal=self.diagonal)
