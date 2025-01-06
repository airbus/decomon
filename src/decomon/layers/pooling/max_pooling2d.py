from decomon.layers import DecomonLayer
from decomon.types import Tensor

import keras
from keras.layers import Layer, Reshape, MaxPooling2D
from keras.models import Sequential
import keras.ops as K
import numpy as np

from decomon.layers.custom.utils import get_affine_lower_bound_max, get_affine_upper_bound_max
from decomon.layers.convolutional.utils import get_toeplitz

from typing import Optional, Any

from decomon.constants import Propagation
from decomon.perturbation_domain import PerturbationDomain
from decomon.utils import memory_limit

from .utils_conv import get_conv_op, get_in_channels, get_backward_layer



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

        # build kernel and depthwise

        if self.affine and (self.propagation == Propagation.BACKWARD or self.affine_shape<memory_limit):
            in_channels:int = get_in_channels(self.layer)
            conv_op, kernel = get_conv_op(self.layer)
            self.conv_op = conv_op
            self.kernel = kernel[:, :, :1]

            # build reshape layer
            input_shape_wo_batch = list(self.layer.input.shape[1:])
            output_shape_wo_batch = list(self.layer.output.shape[1:])

            input_shape_toeplitz = [e for e in input_shape_wo_batch]
            output_shape_toeplitz = [e for e in output_shape_wo_batch]

            if self.layer.data_format == "channels_first":

                in_channel = input_shape_wo_batch[0]
                image_shape = list(self.conv_op(K.zeros([1]+input_shape_wo_batch)).shape[-2:])
                target_shape = [in_channel, -1]+image_shape
                #target_shape = [input_shape_wo_batch[0], -1] + input_shape_wo_batch[1:]
                input_shape_toeplitz[0] = 1
                output_shape_toeplitz[0] = self.conv_op.depth_multiplier
                self.axis = 2
            else:
                raise ValueError()
                target_shape = input_shape_wo_batch + [-1]
                target_shape[-1] = -1
                input_shape_toeplitz[-1] = 1
                output_shape_toeplitz[-1] = self.conv_op.depth_multiplier
                self.axis = -1

            self.reshape_op = Reshape(target_shape)
            self.inner_model = Sequential([self.conv_op, self.reshape_op])
            _ = self.inner_model(self.layer.input)

            self.layer_backward = get_backward_layer(self.conv_op)

            config = self.conv_op.get_config()
            config["kernel_size"] = self.layer.pool_size

            if self.affine and self.propagation == Propagation.BACKWARD:
                # check propagation ...
                if self.fit_memory():

                    self.matrix = get_toeplitz(self.kernel, input_shape_toeplitz, output_shape_toeplitz, config)
                    var = K.eye(in_channels)

                    if self.layer.data_format == "channels_first":
                        var = K.reshape(
                            var,
                            [in_channels]
                            + [1] * (len(input_shape_toeplitz) - 1)
                            + [in_channels]
                            + [1] * len(output_shape_toeplitz),
                        )
                        self.matrix = K.reshape(self.matrix, input_shape_toeplitz + [1] + output_shape_toeplitz)
                    else:
                        raise NotImplementedError()

                    self.matrix = K.expand_dims(self.matrix * var, 0)

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        upper_max = self.reshape_op(self.conv_op(upper))
        lower_max = self.reshape_op(self.conv_op(lower))
        bias = 0.0 * self.layer(lower)

        # compute affine bounds for max(x, axis)
        w_l_max, _ = get_affine_lower_bound_max(lower_max, upper_max, axis=self.axis, keepdims=False)
        w_u_max, b_u = get_affine_upper_bound_max(lower_max, upper_max, axis=self.axis, keepdims=False)

        w_l_max = w_l_max[:, None, None, None]

        w_u_max = w_u_max[:, None, None, None]

        w_u = K.sum(self.matrix * w_u_max, axis=len(self.layer.input.shape) - 1 + self.axis)

        w_l = K.sum(self.matrix * w_l_max, axis=len(self.layer.input.shape) - 1 + self.axis)
        b_l = bias

        return w_l, b_l, w_u, b_u
    

    # override backward propagation
    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        
        if self.layer.data_format=='channels_last':
            return super().backward_affine_propagate(
                output_affine_bounds=output_affine_bounds, input_constant_bounds=input_constant_bounds
            )

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
        
        # compute linear relaxation of the max components
        # for this we need to know the upper and lower bounds at the output of the inner model
        # this model is increasing
        lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)

        if is_output_linear:
            [w_l_wo_batch, b_l_wo_batch, w_u_wo_batch, b_u_wo_batch] = output_affine_bounds
            # add a broadcast dimension
            w_l = K.expand_dims(w_l_wo_batch, 0)  # (1, output_shape, n_out_shape)
            b_l = K.expand_dims(b_l_wo_batch, 0)  # (1, n_out_shape)
            w_u = K.expand_dims(w_u_wo_batch, 0)  # (1, output_shape, n_out_shape)
            b_u = K.expand_dims(b_u_wo_batch, 0)  # (1, n_out_shape)
        else:
            [w_l, b_l, w_u, b_u] = output_affine_bounds

        # split into positive and negative components
        w_u_pos = K.relu(w_u)
        w_l_pos = K.relu(w_l)
        w_u_neg = -K.relu(-w_u)
        w_l_neg = -K.relu(-w_l)

        # expand dims along axis
        w_u_pos_expand = K.expand_dims(w_u_pos, self.axis)
        w_l_pos_expand = K.expand_dims(w_l_pos, self.axis)
        w_u_neg_expand = K.expand_dims(w_u_neg, self.axis)
        w_l_neg_expand = K.expand_dims(w_l_neg, self.axis)

        upper_max = self.reshape_op(self.conv_op(upper))
        lower_max = self.reshape_op(self.conv_op(lower))

        # compute affine bounds for max(x, axis)
        w_l_max, _ = get_affine_lower_bound_max(lower_max, upper_max, axis=self.axis, keepdims=False) #()
        #w_u_max, b_u_max = get_affine_upper_bound_max(lower_max, upper_max, axis=self.axis, keepdims=False)
        w_u_max, b_u_max = get_affine_lower_bound_max(lower_max, upper_max, axis=self.axis, keepdims=False) #()

        n_out_shape = list(b_l.shape[1:])
        w_l_max = K.reshape(w_l_max, [-1]+list(w_l_max.shape[1:])+[1]*len(n_out_shape)) # broadcast dimension
        w_u_max = K.reshape(w_u_max, [-1]+list(w_u_max.shape[1:])+[1]*len(n_out_shape))

        w_u_ = w_u_pos_expand*w_u_max + w_u_neg_expand*w_l_max #(batch, channel_in, depth_mul, width, height, n_out_shape)
        w_l_ = w_l_pos_expand*w_l_max + w_l_neg_expand*w_u_max #(batch, channel_in, depth_mul, width, height, n_out_shape)

        index_output = [ i for i in range(len(b_u_max.shape))][1:]
        b_u_ = K.sum(w_u_pos* K.reshape(b_u_max, [-1]+list(b_u_max.shape[1:])+[1]*len(n_out_shape)), index_output)
        b_l_ = K.sum(w_l_neg* K.reshape(b_u_max, [-1]+list(b_u_max.shape[1:])+[1]*len(n_out_shape)), index_output)

        b_u = b_u + b_u_
        b_l = b_l + b_l_

        # backward on reshape_op
        # channel first
        if self.layer.data_format=='channels_last':
            raise NotImplementedError()
        
        conv_output_shape = list(self.reshape_op.output.shape[2:])
        channel_in = self.layer.input.shape[1]
        n_out_shape_prod = np.prod(n_out_shape)
        w_u_reshape = K.reshape(w_u_, [-1]+conv_output_shape+[n_out_shape_prod]) #(batch*channel_in, depth_mul, width, height, #n_out_shape)
        w_l_reshape = K.reshape(w_l_, [-1]+conv_output_shape+[n_out_shape_prod]) #(batch*channel_in, depth_mul, width, height, #n_out_shape)

        # transpose last dimension to the first axis:
        # (batch*channel_in, depth_mul, width, height, #n_out_shape) -> (batch*channel_in, #n_out_shape, depth_mul, width, height)
        # (0, 1, 2, 3, 4) -> (0, 4, 1, 2, 3)
        w_u_permute = K.transpose(w_u_reshape, (0, 4, 1, 2, 3)) #(batch*channel_in, #n_out_shape, depth_mul, width, height)
        w_l_permute = K.transpose(w_l_reshape, (0, 4, 1, 2, 3)) #(batch*channel_in, #n_out_shape, depth_mul, width, height)

        w_u_backward = K.reshape(w_u_permute, [-1]+conv_output_shape) #(batch*channel_in*#n_out_shape, depth_mul, width, height)
        w_l_backward = K.reshape(w_l_permute, [-1]+conv_output_shape) #(batch*channel_in*#n_out_shape, depth_mul, width, height)

        w_u_conv = self.layer_backward(w_u_backward) #(batch*channel_in*#n_out_shape, 1, in_width, in_height)
        w_l_conv = self.layer_backward(w_l_backward) #(batch*channel_in*#n_out_shape, 1, in_width, in_height)

        # reshape into (batch, channel_in, #n_out_shape, 1, in_width, in_height)
        in_width, in_height = list(w_u_conv.shape[-2:])
        w_u_conv = K.reshape(w_u_conv, [-1, channel_in, n_out_shape_prod, in_width, in_height])
        w_l_conv = K.reshape(w_l_conv, [-1, channel_in, n_out_shape_prod, in_width, in_height])

        # permute dimension
        # [batch, channel_in, n_out_shape_prod, in_width, in_height] -> [batch, channel_in, in_width, in_height, n_out_shape_prod]
        # (0, 1, 2, 3, 4) -> (0, 1, 3, 4, 2)
        w_u_conv = K.transpose(w_u_conv, (0, 1, 3, 4, 2))
        w_l_conv = K.transpose(w_l_conv, (0, 1, 3, 4, 2))

        w_u = K.reshape(w_u_conv, [-1, channel_in, in_width, in_height]+n_out_shape)
        w_l = K.reshape(w_l_conv, [-1, channel_in, in_width, in_height]+n_out_shape)

        return [w_l, b_l, w_u, b_u]
    

        import pdb; pdb.set_trace()





        import pdb; pdb.set_trace()


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
