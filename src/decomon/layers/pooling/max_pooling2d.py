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
    get_batch_multi_dot_repr_for_axis_reduce_weights,
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

    def _backward_affine_propagate_through_linear_block(self, w: Tensor) -> Tensor:
        """Propagate model affine bounds (only upper or lower part) in backward direction.

        We take (lower or upper) affine bounds already propagated through the "max" part
        of the maxpooling layer, and make them propagate through the linear part.

        The computation is the same for upper and lower bounds as we propagate through a linear block.
        Moreover the linear block has no bias so we just propagate the weights.

        Args:
            w: weights

        Returns:
            w_new: propagated weihgts

        """
        linear_block_output_shape_wo_batch = list(self.linear_block.output.shape[1:])
        n_out_linear_block = len(linear_block_output_shape_wo_batch)
        linear_block_input_shape_wo_batch = self.input_shape_wo_batch
        n_in_linear_block = len(linear_block_input_shape_wo_batch)
        model_output_shape_wo_batch = list(w.shape[1 + n_out_linear_block :])
        n_out_model = len(model_output_shape_wo_batch)

        # prepare for backward_layer: layer output shape in last position, and everything else flattened
        # (conversely to decomon convention: (batchsize,) +  layer_output_shape_wo_batch + model_output_shape_wo_batch )
        w_reshaped = K.reshape(
            K.transpose(
                w,
                [0]
                + [n_out_linear_block + 1 + i for i in range(len(model_output_shape_wo_batch))]
                + [i + 1 for i in range(n_out_linear_block)],
            ),
            [-1] + linear_block_output_shape_wo_batch,
        )

        # apply backward_layer
        w_new_reshape_ = self.linear_block_backward(w_reshaped)

        # reshape to get back the decomon convention
        w_new_reshape = K.reshape(
            w_new_reshape_, [-1] + model_output_shape_wo_batch + linear_block_input_shape_wo_batch
        )
        w_new = K.transpose(
            w_new_reshape,
            [0] + [n_out_model + 1 + i for i in range(n_in_linear_block)] + [i + 1 for i in range(n_out_model)],
        )
        return w_new

    def _backward_affine_propagate_through_max(
        self,
        lower: Tensor,
        upper: Tensor,
        w_l: Optional[Tensor],
        w_u: Optional[Tensor],
        is_wo_batch: bool = False,
        is_diagonal: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Propagate model affine bounds (only upper or lower part) in backward direction.

        We take (lower or upper) affine bounds already propagated through the "max" part
        of the maxpooling layer, and make them propagate through the linear part.

        The computation is the same for upper and lower bounds as we propagate through a linear block.
        Moreover the linear block has no bias so we just propagate the weights.

        Args:
            lower: lower constant oracle bound for the max input
            upper: upper constant oracle bound for the max input
            w_l: backward lower weights before propagation through the max
            w_u: backward upper weights before propagation through the max
            is_wo_batch: no batch in weights
            is_diagonal: diagonal weights represented by its diagonal tensor

        Returns:
            w_l_new, b_l_new, w_u_new, b_u_new: propagated backward bounds

        """
        # get non-negative axis index
        if self.axis < 0:
            axis_ = len(lower.shape) + self.axis
        else:
            axis_ = self.axis

        w_l_new, b_l_new = self._backward_affine_propagate_through_max_lower_or_upper(
            lower=lower,
            upper=upper,
            w=w_l,
            axis=axis_,
            compute_lower_bounds=True,
            is_wo_batch=is_wo_batch,
            is_diagonal=is_diagonal,
        )
        w_u_new, b_u_new = self._backward_affine_propagate_through_max_lower_or_upper(
            lower=lower,
            upper=upper,
            w=w_u,
            axis=axis_,
            compute_lower_bounds=False,
            is_wo_batch=is_wo_batch,
            is_diagonal=is_diagonal,
        )
        return w_l_new, b_l_new, w_u_new, b_u_new

    def _backward_affine_propagate_through_max_lower_or_upper(
        self,
        lower: Tensor,
        upper: Tensor,
        w: Optional[Tensor],
        axis: int,
        compute_lower_bounds: bool,
        is_wo_batch: bool = False,
        is_diagonal: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if w is None:  # w == identity
            assert is_diagonal
            if compute_lower_bounds:
                w_new, b_new = get_affine_lower_bound_max_before_reduction(
                    lower=lower, upper=upper, axis=axis, keepdims=False
                )
            else:
                w_new, b_new = get_affine_upper_bound_max_before_reduction(
                    lower=lower, upper=upper, axis=axis, keepdims=False
                )
        else:
            # reshape w (add batch if missing + axis to reduce)
            if is_wo_batch:
                w_e = w[None]
            else:
                w_e = w
            w_e = K.expand_dims(w_e, axis=axis)

            # split into positive and negative parts
            w_pos_e = K.relu(w_e)
            w_neg_e = w_e - w_pos_e

            # broadcast l_c and u_c
            n_model_out = len(w_e.shape) - len(lower.shape)
            if is_diagonal:
                assert n_model_out == 0
            new_shape = lower.shape + (1,) * n_model_out
            lower_e = K.reshape(lower, new_shape)
            upper_e = K.reshape(upper, new_shape)

            # dot(max(z), w+) = sum_{first axes}(max(w+.z))
            lower_0 = lower_e * w_pos_e
            upper_0 = upper_e * w_pos_e
            if compute_lower_bounds:
                w_0, b_0 = get_affine_lower_bound_max_before_reduction(
                    lower=lower_0, upper=upper_0, axis=axis, keepdims=False
                )
            else:
                w_0, b_0 = get_affine_upper_bound_max_before_reduction(
                    lower=lower_0, upper=upper_0, axis=axis, keepdims=False
                )
            # dot(max(z), w-) = - sum_{first axes}(max(-w-.z))
            lower_1 = -upper_e * w_neg_e
            upper_1 = -lower_e * w_neg_e
            if compute_lower_bounds:
                w_1, b_1 = get_affine_upper_bound_max_before_reduction(
                    lower=lower_1, upper=upper_1, axis=axis, keepdims=False
                )
            else:
                w_1, b_1 = get_affine_lower_bound_max_before_reduction(
                    lower=lower_1, upper=upper_1, axis=axis, keepdims=False
                )

            w_new = w_0 - w_1
            b_new = b_0 - b_1

        if is_diagonal:
            # b_new ok
            # w_new nok: we have sum(z.w_new, axis=axis) instead of dot(z, w_new)
            w_new = get_batch_multi_dot_repr_for_axis_reduce_weights(w_new, axis=axis, keepdims=False)
        else:
            # w_new ok (sum over axes will correspond to the dot product)
            # b_new nok:  need to be reduced along first axes
            b_new = K.sum(b_new, axis=list(range(1, 1 + len(self.output_shape_wo_batch))))
        return w_new, b_new

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

        # if diagonal bounds given use method from DecomonLayer
        if len(output_affine_bounds):
            [w_, b_, _, _] = output_affine_bounds
            if is_output_linear:
                is_diagonal = w_.shape == b_.shape
            else:
                is_diagonal = w_.shape[1:] == b_.shape[1:]
        else:
            is_diagonal = True

        if len(output_affine_bounds):
            w_l_out, b_l_out, w_u_out, b_u_out = output_affine_bounds
        else:
            w_l_out, b_l_out, w_u_out, b_u_out = None, None, None, None

        # get oracle bounds for the max inputs
        lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)
        lower_max_input = self.linear_block(lower)
        upper_max_input = self.linear_block(upper)

        # backward propagate through the max
        w_l_max, b_l_max, w_u_max, b_u_max = self._backward_affine_propagate_through_max(
            lower=lower_max_input,
            upper=upper_max_input,
            w_l=w_l_out,
            w_u=w_u_out,
            is_wo_batch=is_output_linear,
            is_diagonal=is_diagonal,
        )

        # backward propagate through the linear block
        w_l = self._backward_affine_propagate_through_linear_block(w_l_max)
        w_u = self._backward_affine_propagate_through_linear_block(w_u_max)

        # add initial bias
        if len(output_affine_bounds):
            b_l = b_l_max + b_l_out  # no bias in the linear block
            b_u = b_u_max + b_u_out  # no bias in the linear block
        else:
            b_l = b_l_max
            b_u = b_u_max

        return (w_l, b_l, w_u, b_u)

        ######
        lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)
        lower_max = self.linear_block(lower)
        upper_max = self.linear_block(upper)

        if self.axis < 0:
            axis_ = len(lower_max.shape) + self.axis
        else:
            axis_ = self.axis

        if is_output_linear:
            output_affine_bounds = [K.expand_dims(e, 0) for e in output_affine_bounds]

        # update w_u_out and w_l_out to have a broadcast dimension at axis
        w_u_out_e = K.expand_dims(w_u_out, axis_)
        w_l_out_e = K.expand_dims(w_l_out, axis_)

        w_u_out_pos_e = K.relu(w_u_out_e)
        w_l_out_pos_e = K.relu(w_l_out_e)
        w_u_out_neg_e = w_u_out_e - w_u_out_pos_e
        w_l_out_neg_e = w_l_out_e - w_l_out_pos_e

        # reshape lower_max and upper_max and update axis if necessary
        n_out = len(w_u_out_e.shape) - len(lower_max.shape)
        expand_shape = [-1] + list(lower_max.shape)[1:] + [1] * n_out
        lower_max_e = K.reshape(lower_max, expand_shape)  # same shape as w_u_out_e
        upper_max_e = K.reshape(upper_max, expand_shape)  # same shape as w_u_out_e

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
