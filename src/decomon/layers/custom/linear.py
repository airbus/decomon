# onnx should use a custom library of keras...
from keras_custom.layers import Linear
from decomon.layers import DecomonLayer
from decomon.types import Tensor
from decomon.layers.utils.affine import get_affine_representation_with_bias

class DecomonLinear(DecomonLayer):
    layer: Linear
    linear=True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:

        return get_affine_representation_with_bias(self.layer, diagonal=self.diagonal)