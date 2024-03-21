import keras.ops as K
from keras.layers import Add

from decomon.layers.merging.base_merge import DecomonMerge
from decomon.types import Tensor


class DecomonAdd(DecomonMerge):
    layer: Add
    linear = True
    diagonal = True

    def get_affine_representation(self) -> tuple[list[Tensor], Tensor]:
        w = [K.ones(input_i.shape[1:]) for input_i in self.keras_layer_input]
        b = K.zeros(self.layer.output.shape[1:])

        return w, b
