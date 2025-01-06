from typing import Any, Optional, Union

from keras.layers import AveragePooling2D
from decomon.layers import DecomonLayer
from keras_custom.backward import get_backward


from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonAveragePooling2D(DecomonLayer):
    layer: AveragePooling2D
    linear = True
    increasing = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:

        return get_affine_representation_wo_bias(self.layer, diagonal=self.diagonal)
    
    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:

        self.layer_backward = get_backward(self.layer, use_bias=False)
        self.layer_backward.trainable=False
        self.layer_backward.built

        super().build(input_shape)   
    
    # override backward propagation
    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        
        if output_affine_bounds is None or len(output_affine_bounds) == 0:
            # no backward affine bounds are propagated; call the affine bounds directly
            return super().backward_affine_propagate(
                output_affine_bounds=output_affine_bounds, input_constant_bounds=input_constant_bounds
            )

        is_output_linear = self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds))
        # if bounds are diagonal, call the affine bounds directly
        # check diagonal
        [w_, b_, _, _] = output_affine_bounds
        if is_output_linear:
            is_diagonal = w_.shape == b_.shape
        else:
            is_diagonal = w_.shape[1:] == b_.shape[1:]

        if is_diagonal:
            return super().backward_affine_propagate(
                output_affine_bounds=output_affine_bounds, input_constant_bounds=input_constant_bounds
            )

        # we optimize the propagation of affine bounds using the backward layers of conv
        # to do so we need to create a new batch of affine bounds
        return self.implicit_linear_backward_affine_propagate(self.layer_backward , output_affine_bounds)
