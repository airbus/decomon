from typing import Optional, Any

from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Layer
from decomon.constants import Propagation
from decomon.perturbation_domain import BoxDomain, PerturbationDomain
from decomon.layers.convolutional.utils import get_toeplitz_from_layer as get_toeplitz
from decomon.layers.utils import get_bias

from decomon.layers.layer import DecomonLayer
from typing import Optional
from decomon.types import Tensor


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

        dico_conv = layer.get_config()
        dico_conv.pop("groups")
        input_shape = list(self.layer.input.shape[1:])
        # update filters to match input, pay attention to data_format
        if layer.data_format == "channels_first":  # better to use enum than raw str
            dico_conv["filters"] = input_shape[0]
        else:
            dico_conv["filters"] = input_shape[-1]

        dico_conv["use_bias"] = False

        self.layer_backward = Conv2DTranspose.from_config(dico_conv)

        self.layer_backward.kernel = layer.kernel
        self.layer_backward.built = True

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:

        b = get_bias(self.layer)
        w = get_toeplitz(self.layer)

        return w, b

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
            return super().backward_affine_propagate(
                output_affine_bounds=output_affine_bounds, input_constant_bounds=input_constant_bounds
            )

        input_shape = list(self.layer.input.shape[1:])
        output_shape = list(self.layer.output.shape[1:])
        bias = self.get_bias()  # retrieve bias component with shape output_shape

        is_output_linear = self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds))

        [w_l, b_l, w_u, b_u] = output_affine_bounds

        if is_output_linear:
            [w_l_wo_batch, b_l_wo_batch, w_u_wo_batch, b_u_wo_batch] = output_affine_bounds
            # add a broadcast dimension
            w_l = K.expand_dims(w_l_wo_batch, 0)  # (batch_size, output_shape, n_out_shape)
            b_l = K.expand_dims(b_l_wo_batch, 0)  # (batch_size, n_out_shape)
            w_u = K.expand_dims(w_u_wo_batch, 0)  # (batch_size, output_shape, n_out_shape)
            b_u = K.expand_dims(b_u_wo_batch, 0)  # (batch_size, n_out_shape)
        else:
            [w_l, b_l, w_u, b_u] = output_affine_bounds

        # step 0: transpose w_l, w_u for shape (batch_size, n_out_shape, output_shape)
        N_output_shape = len(output_shape)
        n_out_shape = list(b_l.shape[1:])
        output_shape_index = [i + 1 for i in range(N_output_shape)]
        n_out_shape_index = [i + N_output_shape for i in range(len(b_l.shape) - 1)]
        index_permute = [0] + n_out_shape_index + output_shape_index
        w_l_permute = K.transpose(w_l, index_permute)  # (batch_size, n_out_shape, output_shape)
        w_u_permute = K.transpose(w_u, index_permute)  # (batch_size, n_out_shape, output_shape)

        w_l_flat = K.reshape(w_l_permute, [-1] + output_shape)
        w_u_flat = K.reshape(w_u_permute, [-1] + output_shape)

        # apply layer + transpose for (batch_size, input_shape, n_out_shape)
        index_post_conv = (
            [0] + [i + 1 + len(n_out_shape) for i in range(len(input_shape))] + [j + 1 for j in range(n_out_shape)]
        )
        w_l_conv = K.transpose(K.reshape(layer_backward(w_l_flat), [-1] + n_out_shape + input_shape), index_post_conv)
        w_u_conv = K.transpose(K.reshape(layer_backward(w_u_flat), [-1] + n_out_shape + input_shape), index_post_conv)

        # w_u*bias (batch_size, output_shape, n_out_shape) * (output_shape,)
        # reshape bias
        bias_ = K.reshape(bias, [-1] + output_shape + [1] * len(input_shape))
        axis_sum = [i + 1 for i in range(len(output_shape))]
        bias_conv_u = K.sum(w_u * bias_, axis_sum)  # (batch_size, n_out_shape)
        bias_conv_l = K.sum(w_u * bias_, axis_sum)  # (batch_size, n_out_shape)

        if is_output_linear:
            return [w_l_conv[0], bias_conv_l[0], w_u_conv[0], bias_conv_u[0]]
        else:
            return [w_l_conv, bias_conv_l, w_u_conv, bias_conv_u]
