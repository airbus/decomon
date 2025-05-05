# define non native class Max
# Decomon Custom for Max(axis...)
from typing import Any

import keras.ops as K
from keras_custom.layers import Max

from decomon.layers.custom.utils import (
    get_affine_lower_bound_max,
    get_affine_upper_bound_max,
)
from decomon.layers.fuse import combine_affine_bounds
from decomon.layers.layer import DecomonLayer
from decomon.types import Tensor


class DecomonMax(DecomonLayer):
    layer: Max
    linear = False
    increasing = True

    def get_affine_bounds_with_linear_block_inputs(
        self, lower_max: Tensor, upper_max: Tensor, axis: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_l, b_l = get_affine_lower_bound_max(lower_max, upper_max, axis=axis, keepdims=False)
        w_u, b_u = get_affine_upper_bound_max(lower_max, upper_max, axis=axis, keepdims=False)

        return (w_l, b_l, w_u, b_u)

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.get_affine_bounds_with_linear_block_inputs(lower_max=lower, upper_max=upper, axis=self.axis)

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor], **kwargs: Any
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Propagate model affine bounds in backward direction.

        By default, this is deduced from `get_affine_bounds()` (or `get_affine_representation()` if `self.linear is True).
        But this could be overridden for better performance. See `DecomonConv2D` for an example.

        Args:
            output_affine_bounds: [w_l, b_l, w_u, b_u]
                partial affine bounds on model output w.r.t underlying keras layer output
            input_constant_bounds: [l_c_in, u_c_in]
                constant oracle bounds on underlying keras layer input

        Returns:
            w_l_new, b_l_new, w_u_new, b_u_new: partial affine bounds on model output w.r.t. underlying keras layer *input*

        If we denote by
          - x: keras model input
          - m(x): keras model output
          - z: underlying keras layer input
          - h(x) = layer(z): output of the underlying keras layer
          - h_i(x) output of the i-th layer
          - w_l_i, b_l_i, w_u_i, b_u_i: current partial linear bounds on model output w.r.t to h_i(x)

        The following inequations are satisfied

            l_c_in <= z <= u_c_in

            Sum_{others layers i}(w_l_i * h_i(x) + b_l_i) +  w_l * h(x) + b_l
              <= m(x)
              <= Sum_{others layers i}(w_u_i * h_i(x) + b_u_i) +  w_u * h(x) + b_u

            Sum_{others layers i}(w_l_i * h_i(x) + b_l_i) +  w_l_new * z + b_l_new
              <= m(x)
              <= Sum_{others layers i}(w_u_i * h_i(x) + b_u_i) +  w_u_new * z + b_u_new

        """
        is_output_linear = self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds))
        lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)

        from_linear_layer = (self.linear, self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds)))

        # if bounds are diagonal, call the affine bounds directly
        # check diagonal
        if len(output_affine_bounds):
            [w_, b_, _, _] = output_affine_bounds
            if is_output_linear:
                is_diagonal = w_.shape == b_.shape
            else:
                is_diagonal = w_.shape[1:] == b_.shape[1:]
        else:
            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower=lower, upper=upper)
            return (w_l, b_l, w_u, b_u)

        if is_diagonal:
            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower=lower, upper=upper)
            layer_affine_bounds = [w_l, b_l, w_u, b_u]

        if is_diagonal or not len(output_affine_bounds):
            diagonal = (
                self.inputs_outputs_spec.is_diagonal_bounds(layer_affine_bounds),
                self.inputs_outputs_spec.is_diagonal_bounds(output_affine_bounds),
            )

            return combine_affine_bounds(
                affine_bounds_1=layer_affine_bounds,
                affine_bounds_2=output_affine_bounds,
                from_linear_layer=from_linear_layer,
                diagonal=diagonal,
            )

        #######

        if is_output_linear:
            output_affine_bounds = [K.expand_dims(e, 0) for e in output_affine_bounds]

        [w_l_out, b_l_out, w_u_out, b_u_out] = output_affine_bounds

        lower_max = self.linear_block(lower)
        upper_max = self.linear_block(upper)

        # update w_u_out and w_l_out to have a broadcast dimension at axis
        w_u_out_e = K.expand_dims(w_u_out, self.axis)
        w_l_out_e = K.expand_dims(w_l_out, self.axis)

        w_u_out_pos_e = K.relu(w_u_out_e)
        w_l_out_pos_e = K.relu(w_l_out_e)
        w_u_out_neg_e = w_u_out_e - w_u_out_pos_e
        w_l_out_neg_e = w_l_out_e - w_l_out_pos_e

        # reshape lower_max and upper_max and update axis if necessary
        n_out = len(w_u_out_e.shape) - len(lower_max.shape)
        expand_shape = [-1] + list(lower_max.shape)[1:] + [1] * n_out
        lower_max_e = K.reshape(lower_max, expand_shape)  # same shape as w_u_out_e
        upper_max_e = K.reshape(upper_max, expand_shape)  # same shape as w_u_out_e

        if self.axis == -1:
            axis_ = len(lower_max.shape) - 1
        else:
            axis_ = self.axis

        lower_max_u_0 = lower_max_e * w_u_out_pos_e
        upper_max_u_0 = upper_max_e * w_u_out_pos_e
        _, _, w_u_0, b_u_0 = self.get_affine_bounds_with_linear_block_inputs(
            lower_max=lower_max_u_0, upper_max=upper_max_u_0, axis=axis_
        )

        lower_max_u_1 = -lower_max_e * w_u_out_neg_e
        upper_max_u_1 = -upper_max_e * w_u_out_neg_e
        w_l_1, b_l_1, _, _ = self.get_affine_bounds_with_linear_block_inputs(
            lower_max=lower_max_u_1, upper_max=upper_max_u_1, axis=axis_
        )

        w_u = w_u_0 - w_l_1
        b_u = b_u_0 - b_l_1 + b_u_out

        #### lower bound
        lower_max_l_0 = lower_max_e * w_l_out_pos_e
        upper_max_l_0 = upper_max_e * w_l_out_pos_e
        w_l_0, b_l_0, _, _ = self.get_affine_bounds_with_linear_block_inputs(
            lower_max=lower_max_l_0, upper_max=upper_max_l_0, axis=axis_
        )
        lower_max_l_1 = -lower_max_e * w_l_out_neg_e
        upper_max_l_1 = -upper_max_e * w_l_out_neg_e
        _, _, w_u_1, b_u_1 = self.get_affine_bounds_with_linear_block_inputs(
            lower_max=lower_max_l_1, upper_max=upper_max_l_1, axis=axis_
        )

        w_l = w_l_0 - w_u_1
        b_l = b_l_0 - b_u_1 + b_l_out

        return (w_l, b_l, w_u, b_u)
