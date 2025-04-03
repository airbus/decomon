from keras.layers import GlobalAveragePooling3D  # type:ignore

from decomon.layers import DecomonLayer
from decomon.layers.utils import get_affine_representation_wo_bias
from decomon.types import Tensor


class DecomonGlobalAveragePooling3D(DecomonLayer):
    layer: GlobalAveragePooling3D
    linear = True
    increasing = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        return get_affine_representation_wo_bias(self.layer, diagonal=self.diagonal)
