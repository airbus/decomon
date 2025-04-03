from typing import Any, Optional

import keras.ops as K  # type:ignore
import numpy as np  # type:ignore
from jacobinet.layers.convert import get_backward  # type:ignore
from jacobinet.layers.pooling.utils_max import get_linear_block_max  # type:ignore
from keras.layers import Layer, MaxPooling2D, Reshape  # type:ignore
from keras.models import Sequential  # type:ignore

from decomon.constants import Propagation
from decomon.layers import DecomonLayer
from decomon.layers.convolutional.utils import get_toeplitz
from decomon.layers.custom.utils import (
    get_affine_lower_bound_max,
    get_affine_upper_bound_max,
)
from decomon.layers.fuse import combine_affine_bounds
from decomon.layers.utils.affine import get_bias
from decomon.perturbation_domain import PerturbationDomain
from decomon.types import Tensor
from decomon.utils import memory_limit

from .utils_conv import get_conv_op, get_in_channels


class DecomonMaxPooling2D(DecomonLayer):
    layer: MaxPooling2D
    linear: False
    increasing = True

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
        self, lower_max: Tensor, upper_max: Tensor, axis=int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_l_max, b_l_max = get_affine_lower_bound_max(lower_max, upper_max, axis=axis, keepdims=False)
        w_u_max, b_u_max = get_affine_upper_bound_max(lower_max, upper_max, axis=axis, keepdims=False)

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

        return [w_l, b_l, w_u, b_u]

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        lower_max = self.linear_block(lower)
        upper_max = self.linear_block(upper)

        return self.get_affine_bounds_with_linear_block_inputs(lower_max=lower_max, upper_max=upper_max, axis=self.axis)

    def call_forward(
        self,
        affine_bounds_to_propagate: list[Tensor],
        input_bounds_to_propagate: list[Tensor],
        perturbation_domain_inputs: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Propagate forward affine and constant bounds through the layer.

        Args:
            affine_bounds_to_propagate: affine bounds on keras layer input w.r.t model input.
            Can be empty if not in affine mode.
            Can also be empty in case of identity affine bounds => we simply return layer affine bounds.
            input_bounds_to_propagate: ibp constant bounds on keras layer input. Can be empty if not in ibp mode.
            perturbation_domain_inputs: perturbation domain input, wrapped in a list. Necessary only in affine mode, else empty.

        Returns:
            output_affine_bounds, output_constant_bounds: affine and constant bounds on the underlying keras layer output

        Note:
            In hybrid case (ibp+affine), the constant bounds are assumed to be already tight in input, and we will return
            the tighter constant bounds in output. This means that
            - for the output: we take the tighter constant bounds between the ibp ones and the ones deduced
                from the affine bounds given the considered perturbation domain, on the output.
            - for the input: we do not need it, as it should already have been taken care of in the previous layer

        """

        # IBP: interval bounds propragation
        if self.ibp:
            lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_bounds_to_propagate)
            output_constant_bounds = list(self.forward_ibp_propagate(lower=lower, upper=upper))
        else:
            output_constant_bounds = []

        # Affine bounds propagation
        if self.affine:
            # forward propagation
            output_affine_bounds = list(
                self.forward_affine_propagate(
                    input_affine_bounds=affine_bounds_to_propagate, input_constant_bounds=perturbation_domain_inputs
                )
            )
        else:
            output_affine_bounds = []

        # Tighten constant bounds in hybrid mode (ibp+affine)
        if self.ibp and self.affine:
            if len(perturbation_domain_inputs) == 0:
                raise RuntimeError("keras model input is necessary for call_forward() in affine mode.")
            x = perturbation_domain_inputs[0]
            l_ibp, u_ibp = output_constant_bounds
            w_l, b_l, w_u, b_u = output_affine_bounds
            from_linear = self.linear and self.inputs_outputs_spec.is_wo_batch_bounds(
                affine_bounds=affine_bounds_to_propagate
            )
            l_affine = self.perturbation_domain.get_lower(x, w_l, b_l, missing_batchsize=from_linear)
            u_affine = self.perturbation_domain.get_upper(x, w_u, b_u, missing_batchsize=from_linear)
            u = K.minimum(u_ibp, u_affine)
            l = K.maximum(l_ibp, l_affine)
            output_constant_bounds = [l, u]

        return output_affine_bounds, output_constant_bounds

    def forward_affine_propagate(
        self, input_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
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

        # get input_constant_bounds after linear_block
        if is_from_linear:
            # add broadcast dimension for batch
            linear_bounds = [K.expand_dims(e, 0) for e in linear_bounds]
        w_l_out, b_l_out, w_u_out, b_u_out = linear_bounds
        dim = np.prod([self.layer.pool_size])
        broadcast_shape = [1] + [1] * len(self.model_input_shape) + [1] * len(layer_output_shape_wo_batchsize)

        if self.axis > 0:
            broadcast_shape[len(self.model_input_shape) + self.axis] = dim
        else:
            broadcast_shape[self.axis] = dim

        x = input_constant_bounds[0]
        lower = self.perturbation_domain.get_lower(x, w_l_out, b_l_out, missing_batchsize=False)
        upper = self.perturbation_domain.get_upper(x, w_u_out, b_u_out, missing_batchsize=False)

        w_u_max, b_u_max = get_affine_upper_bound_max(lower=lower, upper=upper, axis=self.axis, keepdims=False)
        w_l_max, b_l_max = get_affine_lower_bound_max(lower=lower, upper=upper, axis=self.axis, keepdims=False)

        N = len(self.model_input_shape)
        w_u_max_ = K.reshape(w_u_max, [-1] + [1] * N + list(w_u_max.shape[1:]))
        w_l_max_ = K.reshape(w_l_max, [-1] + [1] * N + list(w_u_max.shape[1:]))

        if self.axis > 0:
            axis_ = self.axis + N
        else:
            axis_ = self.axis
        w_u_out = K.sum(w_u_max_ * w_u_out, axis_)
        w_l_out = K.sum(w_l_max_ * w_l_out, axis_)
        b_u_out = K.sum(K.sum(w_u_max_ * b_u_out, axis=tuple(np.arange(1, N + 1))), self.axis) + b_u_max
        b_l_out = K.sum(K.sum(w_l_max_ * b_l_out, axis=tuple(np.arange(1, N + 1))), self.axis) + b_l_max

        return [w_l_out, b_l_out, w_u_out, b_u_out]

    def backward_affine_propagate_single_channel(
        self, lower, upper, output_affine_bounds: list[Tensor]
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
            layer_affine_bounds = [w_l, b_l, w_u, b_u]
            return layer_affine_bounds

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

        return [w_l, b_l, w_u, b_u]

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
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
            layer_affine_bounds = [w_l, b_l, w_u, b_u]
            return layer_affine_bounds

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

        return [w_l, b_l, w_u, b_u]
