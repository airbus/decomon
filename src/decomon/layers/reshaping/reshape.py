from keras.layers import Reshape
from decomon.layers import DecomonLayer
import keras.ops as K

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonReshape(DecomonLayer):
    layer: Reshape
    linear = True
    increasing = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:

        return get_affine_representation_wo_bias(self.layer, diagonal=self.diagonal)
    
    # override backward propagation
    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        if output_affine_bounds is None or len(output_affine_bounds) == 0:
            return super().backward_affine_propagate(
                output_affine_bounds=output_affine_bounds, input_constant_bounds=input_constant_bounds
            )

        input_shape = list(self.layer.input.shape[1:]) # (in_channel, in_w, in_h) if data_format=='channel_first'

        is_output_linear = self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds))

        [w_l, b_l, w_u, b_u] = output_affine_bounds

        # step 0: transpose w_l, w_u for shape (batch_size, n_out_shape, output_shape)
        if is_output_linear:
            n_out_shape = list(b_l.shape)
            target_shape = input_shape+n_out_shape
        else:
            n_out_shape = list(b_l.shape[1:])
            target_shape = [-1]+input_shape+n_out_shape

        w_l = K.reshape(w_l, target_shape)
        w_u = K.reshape(w_u, target_shape)

        return [w_l, b_l, w_u, b_u]
