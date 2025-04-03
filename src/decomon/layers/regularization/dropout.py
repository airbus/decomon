# onnx should use a custom library of keras...
from keras.layers import Dropout  # type:ignore

from decomon.layers import DecomonLayer
from decomon.layers.utils.affine import get_affine_representation_wo_bias
from decomon.types import Tensor


class DecomonDropout(DecomonLayer):
    layer: Dropout
    linear = True
    diagonal = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        return get_affine_representation_wo_bias(self.layer, diagonal=self.diagonal)
