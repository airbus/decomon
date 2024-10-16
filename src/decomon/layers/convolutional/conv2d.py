from typing import Optional, Any

from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Layer
import keras.ops as K
from decomon.constants import Propagation
from decomon.perturbation_domain import BoxDomain, PerturbationDomain
from decomon.layers.convolutional.utils import get_toeplitz_from_layer as get_toeplitz
from decomon.layers.utils import get_bias


from decomon.layers.layer import DecomonLayer
from typing import Optional
from decomon.types import Tensor

import numpy as np


def get_backward_layer(layer: Conv2D) -> Layer:

    dico_conv = layer.get_config()
    dico_conv.pop("groups")
    input_shape = list(layer.input.shape[1:])
    # update filters to match input, pay attention to data_format
    if layer.data_format == "channels_first":  # better to use enum than raw str
        dico_conv["filters"] = input_shape[0]
    else:
        dico_conv["filters"] = input_shape[-1]

    dico_conv["use_bias"] = False

    layer_backward = Conv2DTranspose.from_config(dico_conv)
    layer_backward.kernel = layer.kernel
    layer_backward.built = True

    return layer_backward


class DecomonConv2D(DecomonLayer):

    layer: Conv2D
    linear = True

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

        layer_backward = get_backward_layer(layer)

        super().__init__(
            layer=layer,
            perturbation_domain=perturbation_domain,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            layer_backward=layer_backward,
            **kwargs,
        )

        self.b = get_bias(layer)

        if self.affine and self.propagation == Propagation.BACKWARD:
            # check propagation ...
            self.w = get_toeplitz(self.layer)

        # conv_pos = Conv2D.from_config(self.layer.get_config())
        # conv_pos._kernel = K.relu(self.layer.kernel)

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:

        # b = get_bias(self.layer)
        # w = get_toeplitz(self.layer)

        return self.w, self.b

    # override backward propagation
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

        input_shape = list(self.layer.input.shape[1:])  # (in_channel, in_w, in_h) if data_format=='channel_first'
        output_shape = list(self.layer.output.shape[1:])  # (out_channel, out_w, out_h)

        if is_output_linear:
            [w_l_wo_batch, b_l_wo_batch, w_u_wo_batch, b_u_wo_batch] = output_affine_bounds
            # add a broadcast dimension
            w_l = K.expand_dims(w_l_wo_batch, 0)  # (1, output_shape, n_out_shape)
            b_l = K.expand_dims(b_l_wo_batch, 0)  # (1, n_out_shape)
            w_u = K.expand_dims(w_u_wo_batch, 0)  # (1, output_shape, n_out_shape)
            b_u = K.expand_dims(b_u_wo_batch, 0)  # (1, n_out_shape)
        else:
            [w_l, b_l, w_u, b_u] = output_affine_bounds

        n_out_shape = list(b_l.shape[1:])
        n_out_shape_flat = int(np.prod(n_out_shape))

        w_l_flat_0 = K.reshape(w_l, [-1] + output_shape + [n_out_shape_flat])  # (batch, output_shape, n_out_flat)
        w_u_flat_0 = K.reshape(w_u, [-1] + output_shape + [n_out_shape_flat])  # (batch, output_shape, n_out_flat)

        # permute dimension
        N_output_shape = len(output_shape)  # number of dimensions without batch size
        output_shape_index = [i + 1 for i in range(N_output_shape)]

        w_l_flat = K.transpose(
            w_l_flat_0, [0, N_output_shape + 1] + output_shape_index
        )  # (batch, n_out_flat, output_shape)
        w_u_flat = K.transpose(
            w_u_flat_0, [0, N_output_shape + 1] + output_shape_index
        )  # (batch, n_out_flat, output_shape)

        w_l_flat_ = K.reshape(w_l_flat, [-1] + output_shape)  # (batch*n_out_flat, output_shape)
        w_u_flat_ = K.reshape(w_u_flat, [-1] + output_shape)  # (batch*n_out_flat, output_shape)

        # apply backward layer
        w_l_conv = self.layer_backward(w_l_flat_)  # (batch*n_out_flat, input_shape)
        w_u_conv = self.layer_backward(w_u_flat_)  # (batch*n_out_flat, input_shape)

        # reshape to (batch, n_out_flat, input_shape)
        w_l_conv = K.reshape(w_l_conv, [-1, n_out_shape_flat] + input_shape)
        w_u_conv = K.reshape(w_u_conv, [-1, n_out_shape_flat] + input_shape)

        # permute dimensions: (batch, input_shape, n_out_flat)
        # (0, 1, 2, 3, 4) -> (0, 2, 3, 4, 1)
        input_shape_index = [0] + [i + 2 for i in range(len(input_shape))] + [1]
        w_l_conv = K.transpose(w_l_conv, input_shape_index)
        w_u_conv = K.transpose(w_u_conv, input_shape_index)

        # reshape to (batch, input_shape, n_out)
        w_l_conv = K.reshape(w_l_conv, [-1] + input_shape + n_out_shape)
        w_u_conv = K.reshape(w_u_conv, [-1] + input_shape + n_out_shape)

        # convert bias to an additive term
        bias = get_bias(self.layer)  # retrieve bias component with shape output_shape
        # w_u*bias (batch_size, output_shape, n_out_shape) * (output_shape,)
        # reshape bias
        bias_ = K.reshape(bias, [-1] + output_shape + [1] * len(n_out_shape))
        # axis_sum = [i + 1 for i in range(len(output_shape))]
        axis_sum = output_shape_index
        bias_conv_u = K.sum(w_u * bias_, axis_sum) + b_u  # (batch_size, n_out_shape)
        bias_conv_l = K.sum(w_l * bias_, axis_sum) + b_l  # (batch_size, n_out_shape)

        if is_output_linear:
            output = [w_l_conv[0], bias_conv_l[0], w_u_conv[0], bias_conv_u[0]]
        else:
            output = [w_l_conv, bias_conv_l, w_u_conv, bias_conv_u]

        return output
