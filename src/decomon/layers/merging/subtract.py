import keras.ops as K
from keras.layers import Subtract

from decomon.layers.merging.base_merge import DecomonMerge
from decomon.types import Tensor


class DecomonSubtract(DecomonMerge):
    layer: Subtract
    linear = True
    diagonal = True

    def get_affine_representation(self) -> tuple[list[Tensor], Tensor]:

        w = [K.ones(input_i.shape[1:]) for input_i in self.keras_layer_input]
        w[1] *= -1
        b = K.zeros(self.layer.output.shape[1:])

        return w, b
