from typing import Any, Optional

import keras.ops as K
import numpy as np
from jacobinet.layers.convert import get_backward
from jacobinet.layers.pooling.utils_max import get_linear_block_max
from keras.layers import Layer, MaxPooling2D

from decomon.constants import Propagation
from decomon.layers import DecomonLayer
from decomon.layers.custom.reduce.utils import (
    get_affine_lower_bound_max_before_reduction,
    get_affine_upper_bound_max_before_reduction,
)
from decomon.layers.fuse import combine_affine_bounds
from decomon.perturbation_domain import PerturbationDomain
from decomon.types import Tensor


class DecomonMaxPooling2D(DecomonLayer):
    layer: MaxPooling2D
    linear = False
    increasing = True
    use_bias = False
    skip_forward_oracle = True  # forward oracle performed in overriden forward_affine_propagate()

    def __init__(
        self,
        layer: Layer,
        perturbation_domain: Optional[PerturbationDomain] = None,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        model_input_shape: Optional[tuple[int, ...]] = None,
        model_output_shape: Optional[tuple[int, ...]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            perturbation_domain=perturbation_domain,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            **kwargs,
        )

        # from get_backward we only retrieve the backward of the linear components of MaxPooling
        layer_backward_maxpool = get_backward(self.layer)
        # from get_backward we only retrieve the backward of the linear components of MaxPooling

        self.linear_block = layer_backward_maxpool.linear_block
        self.linear_block_backward = layer_backward_maxpool.linear_block_backward

        input_dim_wo_batch_wo_channel = list(self.layer.input.shape[1:])
        if self.layer.data_format == "channels_last":
            input_dim_wo_batch_wo_channel[-1] = 1
        else:
            input_dim_wo_batch_wo_channel[0] = 1
        self.linear_block_backward_single_channel = get_linear_block_max(self.layer, input_dim_wo_batch_wo_channel)
        self.axis = layer_backward_maxpool.axis

    def get_affine_bounds_with_linear_block_inputs(
        self, lower_max: Tensor, upper_max: Tensor, axis: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_l_max, b_l_max = get_affine_lower_bound_max_before_reduction(lower_max, upper_max, axis=axis, keepdims=False)
        w_u_max, b_u_max = get_affine_upper_bound_max_before_reduction(lower_max, upper_max, axis=axis, keepdims=False)

        output_shape_wo_batch = list(self.linear_block.output.shape[1:])

        # derive n_out
        N_out = len(output_shape_wo_batch)
        N_in = self.input_shape_wo_batch
        n_out = list(w_l_max.shape[1:])[N_out:]

        w_l_max_reshape = K.transpose(
            w_l_max, [0] + [N_out + 1 + i for i in range(len(n_out))] + [i + 1 for i in range(N_out)]
        )
        w_u_max_reshape = K.transpose(
            w_u_max, [0] + [N_out + 1 + i for i in range(len(n_out))] + [i + 1 for i in range(N_out)]
        )

        w_l_max_reshape_ = K.reshape(w_l_max_reshape, [-1] + output_shape_wo_batch)
        w_u_max_reshape_ = K.reshape(w_u_max_reshape, [-1] + output_shape_wo_batch)

        # because max is increasing, we know that w_u_max and w_l_max contain only positive values
        # thus we do not need to split to follow backward propagation rule
        # note that we should do this optimization in the general case
        n_out_dim = list(w_l_max_reshape.shape[1 : len(n_out) + 1])
        w_l_reshape_ = self.linear_block_backward(w_l_max_reshape_)
        w_u_reshape_ = self.linear_block_backward(w_u_max_reshape_)

        # reshape again
        w_l_reshape = K.reshape(w_l_reshape_, [-1] + n_out_dim + list(w_l_reshape_.shape[1:]))
        w_u_reshape = K.reshape(w_u_reshape_, [-1] + n_out_dim + list(w_u_reshape_.shape[1:]))

        w_l = K.transpose(
            w_l_reshape, [0] + [len(n_out) + 1 + i for i in range(len(N_in))] + [i + 1 for i in range(len(n_out))]
        )
        w_u = K.transpose(
            w_u_reshape, [0] + [len(n_out) + 1 + i for i in range(len(N_in))] + [i + 1 for i in range(len(n_out))]
        )

        if len(n_out):
            b_u = K.sum(b_u_max, axis=[i + 1 for i in range(N_out - 1)])
            b_l = K.sum(b_l_max, axis=[i + 1 for i in range(N_out - 1)])
        else:
            b_u = b_u_max
            b_l = b_l_max

        return (w_l, b_l, w_u, b_u)

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        lower_max = self.linear_block(lower)
        upper_max = self.linear_block(upper)

        return self.get_affine_bounds_with_linear_block_inputs(lower_max=lower_max, upper_max=upper_max, axis=self.axis)

    def forward_affine_propagate(
        self,
        input_affine_bounds: list[Tensor],
        input_constant_bounds: list[Tensor],
        perturbation_domain_inputs: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        is_from_linear = self.inputs_outputs_spec.is_wo_batch_bounds(input_affine_bounds)
        layer_output_shape_wo_batchsize = list(self.linear_block.output.shape[1:])
        linear_bounds = self._forward_affine_propagate_linear(
            self.linear_block,
            None,
            None,
            self.layer_input_shape_wo_batchsize,
            layer_output_shape_wo_batchsize,
            input_affine_bounds,
        )

        w_l_out, b_l_out, w_u_out, b_u_out = linear_bounds

        # get constant bounds after linear_block
        x = perturbation_domain_inputs[0]
        lower_max_input_affine = self.perturbation_domain.get_lower(
            x, w_l_out, b_l_out, missing_batchsize=is_from_linear
        )
        upper_max_input_affine = self.perturbation_domain.get_upper(
            x, w_u_out, b_u_out, missing_batchsize=is_from_linear
        )
        if self.ibp:
            # tighten with ibp propagation
            lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)
            lower_max_input_ibp = self.linear_block(lower)
            upper_max_input_ibp = self.linear_block(upper)
            lower_max_input = K.maximum(lower_max_input_ibp, lower_max_input_affine)
            upper_max_input = K.minimum(upper_max_input_ibp, upper_max_input_affine)
        else:
            lower_max_input = lower_max_input_affine
            upper_max_input = upper_max_input_affine

        w_u_max, b_u_max = get_affine_upper_bound_max_before_reduction(
            lower=lower_max_input, upper=upper_max_input, axis=self.axis, keepdims=False
        )
        w_l_max, b_l_max = get_affine_lower_bound_max_before_reduction(
            lower=lower_max_input, upper=upper_max_input, axis=self.axis, keepdims=False
        )

        N = len(self.model_input_shape)
        w_u_max_ = K.reshape(w_u_max, [-1] + [1] * N + list(w_u_max.shape[1:]))
        w_l_max_ = K.reshape(w_l_max, [-1] + [1] * N + list(w_u_max.shape[1:]))

        if is_from_linear:
            # add broadcast dimension for batch
            w_l_out = w_l_out[None]
            w_u_out = w_u_out[None]
            b_u_out = b_u_out[None]
            b_l_out = b_l_out[None]

        if self.axis > 0:
            axis_ = self.axis + N
        else:
            axis_ = self.axis
        w_u_out = K.sum(w_u_max_ * w_u_out, axis_)
        w_l_out = K.sum(w_l_max_ * w_l_out, axis_)
        b_u_out = K.sum(K.sum(w_u_max_ * b_u_out, axis=tuple(np.arange(1, N + 1))), self.axis) + b_u_max
        b_l_out = K.sum(K.sum(w_l_max_ * b_l_out, axis=tuple(np.arange(1, N + 1))), self.axis) + b_l_max

        return (w_l_out, b_l_out, w_u_out, b_u_out)

    def backward_affine_propagate_single_channel(
        self, lower: Tensor, upper: Tensor, output_affine_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        is_output_linear = self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds))

        from_linear_layer = (self.linear, self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds)))

        # if bounds are diagonal, call the affine bounds directly
        # check diagonal
        [w_, b_, _, _] = output_affine_bounds

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

        # HERE
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

        lower_max_u_1 = -upper_max_e * w_u_out_neg_e
        upper_max_u_1 = -lower_max_e * w_u_out_neg_e
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
        lower_max_l_1 = -upper_max_e * w_l_out_neg_e
        upper_max_l_1 = -lower_max_e * w_l_out_neg_e
        _, _, w_u_1, b_u_1 = self.get_affine_bounds_with_linear_block_inputs(
            lower_max=lower_max_l_1, upper_max=upper_max_l_1, axis=axis_
        )

        w_l = w_l_0 - w_u_1
        b_l = b_l_0 - b_u_1 + b_l_out

        return (w_l, b_l, w_u, b_u)

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

        if self.layer.data_format == "channels_first":
            channel = lower.shape[1]
            axis = 1
        else:
            channel = lower.shape[-1]
            axis = 3

        from_linear_layer = (self.linear, self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds)))

        # if bounds are diagonal, call the affine bounds directly
        # check diagonal
        [w_, b_, _, _] = output_affine_bounds

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

        w_l_out, b_l_out, w_u_out, b_u_out = output_affine_bounds

        lower_max = self.linear_block(lower)
        upper_max = self.linear_block(upper)

        # update w_u_out and w_l_out to have a broadcast dimension at axis
        if self.axis == -1:
            w_u_out_e = K.expand_dims(w_u_out, -2)
            w_l_out_e = K.expand_dims(w_l_out, -2)
        else:
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

        lower_max_u_1 = -upper_max_e * w_u_out_neg_e
        upper_max_u_1 = -lower_max_e * w_u_out_neg_e
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
        lower_max_l_1 = -upper_max_e * w_l_out_neg_e
        upper_max_l_1 = -lower_max_e * w_l_out_neg_e
        _, _, w_u_1, b_u_1 = self.get_affine_bounds_with_linear_block_inputs(
            lower_max=lower_max_l_1, upper_max=upper_max_l_1, axis=axis_
        )

        w_l = w_l_0 - w_u_1
        b_l = b_l_0 - b_u_1 + b_l_out

        return (w_l, b_l, w_u, b_u)
