import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d
from tensorflow.keras.layers import InputSpec, MaxPooling2D

from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.utils import max_
from decomon.utils import get_lower, get_upper


# step 1: compute the maximum
class DecomonMaxPooling2D(MaxPooling2D, DecomonLayer):
    """LiRPA implementation of MaxPooling2D layers.
    See Keras official documentation for further details on the MaxPooling2D operator

    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        mode=ForwardMode.HYBRID,
        fast=True,
        **kwargs,
    ):

        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            mode=mode,
            fast=fast,
            **kwargs,
        )

        if self.mode == ForwardMode.IBP:
            self.input_spec = [
                InputSpec(ndim=4),  # u
                InputSpec(ndim=4),  # l
            ]
        elif self.mode == ForwardMode.AFFINE:
            self.input_spec = [
                InputSpec(min_ndim=2),  # x
                InputSpec(ndim=5),  # w_u
                InputSpec(ndim=4),  # b_u
                InputSpec(ndim=5),  # w_l
                InputSpec(ndim=4),
            ]  # b_l
        elif self.mode == ForwardMode.HYBRID:
            self.input_spec = [
                InputSpec(min_ndim=2),  # x
                InputSpec(ndim=4),  # u
                InputSpec(ndim=5),  # w_u
                InputSpec(ndim=4),  # b_u
                InputSpec(ndim=4),  # l
                InputSpec(ndim=5),  # w_l
                InputSpec(ndim=4),
            ]  # b_l
        else:
            raise ValueError(f"Unknown mode {self.mode}.")

        if self.dc_decomp:
            self.input_spec += [InputSpec(ndim=4), InputSpec(ndim=4)]

        # express maxpooling with convolutions
        self.filters = np.zeros((pool_size[0], pool_size[1], 1, np.prod(pool_size)), dtype=self.dtype)
        for i in range(pool_size[0]):
            for j in range(pool_size[1]):
                self.filters[i, j, 0, i * pool_size[0] + j] = 1

        def conv_(x):

            if self.data_format in [None, "channels_last"]:
                return K.cast(
                    K.expand_dims(
                        conv2d(
                            x,
                            self.filters,
                            strides=strides,
                            padding=padding,
                            data_format=data_format,
                        ),
                        -2,
                    ),
                    self.dtype,
                )
            else:
                return K.cast(
                    K.expand_dims(
                        conv2d(
                            x,
                            self.filters,
                            strides=strides,
                            padding=padding,
                            data_format=data_format,
                        ),
                        1,
                    ),
                    self.dtype,
                )

        self.internal_op = conv_

    def compute_output_shape(self, input_shape):
        """
        Args:
            input_shape

        Returns:

        """

        if self.grad_bounds:
            raise NotImplementedError()

        output_shape_ = super().compute_output_shape(input_shape[0])
        input_dim = input_shape[-1][1]
        if self.mode == ForwardMode.IBP:
            output_shape = [output_shape_] * 2
        elif self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            x_shape = input_shape[1]
            w_shape_ = tuple([output_shape_[0], input_dim] + list(output_shape_)[1:])
            if self.mode == ForwardMode.AFFINE:
                output_shape = [x_shape] + [w_shape_, output_shape_] * 2
            else:
                output_shape = [x_shape] + [output_shape_, w_shape_, output_shape_] * 2
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output_shape += [output_shape_] * 2

        return output_shape

    def _pooling_function_fast(self, inputs, pool_size, strides, padding, data_format, mode):

        if self.dc_decomp:
            raise NotImplementedError()

        if mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
        elif mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]
        elif mode == ForwardMode.IBP:
            u_c, l_c = inputs[: self.nb_tensors]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            l_c_ = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")
            u_c_ = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")

        if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:

            if mode in ForwardMode.AFFINE:
                u_c = get_upper(x_0, w_u, b_u)
                l_c = get_lower(x_0, w_l, b_l)

                l_c_ = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")
                u_c_ = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")

            n_in = x_0.shape[-1]
            w_u_ = K.concatenate([0 * K.expand_dims(u_c_, 1)] * n_in, 1)
            w_l_ = w_u_
            b_u_ = u_c_
            b_l_ = l_c_

        if mode == ForwardMode.IBP:
            output = [u_c_, l_c_]
        elif mode == ForwardMode.HYBRID:
            output = [
                x_0,
                u_c_,
                w_u_,
                b_u_,
                l_c_,
                w_l_,
                b_l_,
            ]
        elif mode == ForwardMode.AFFINE:
            output = [
                x_0,
                w_u_,
                b_u_,
                w_l_,
                b_l_,
            ]
        else:
            raise ValueError(f"Unknown mode {mode}")
        return output

    def _pooling_function_not_fast(self, inputs, pool_size, strides, padding, data_format, mode):
        nb_tensors = self.nb_tensors
        if self.dc_decomp:
            h, g = inputs[-2:]
            nb_tensors -= 2

        if mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        elif mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
        elif mode == ForwardMode.IBP:
            u_c, l_c = inputs[:nb_tensors]
        else:
            raise ValueError(f"Unknown mode {mode}")

        input_shape = K.int_shape(inputs[-1])

        if self.dc_decomp:
            h_list = K.concatenate(
                [self.internal_op(elem) for elem in tf.split(h, input_shape[-1], -1)],
                -2,
            )
            g_list = K.concatenate(
                [self.internal_op(elem) for elem in tf.split(g, input_shape[-1], -1)],
                -2,
            )

        if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(u_c, input_shape[-1], -1)], -2)
            l_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(l_c, input_shape[-1], -1)], -2)

        if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:

            b_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_u, input_shape[-1], -1)], -2)
            b_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_l, input_shape[-1], -1)], -2)
            w_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_u, input_shape[-1], -1)], -2)
            w_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_l, input_shape[-1], -1)], -2)

        if mode == ForwardMode.IBP:
            output_list = [u_c_list, l_c_list]
        elif mode == ForwardMode.HYBRID:
            output_list = [
                x_0,
                u_c_list,
                w_u_list,
                b_u_list,
                l_c_list,
                w_l_list,
                b_l_list,
            ]
        elif mode == ForwardMode.AFFINE:
            output_list = [
                x_0,
                w_u_list,
                b_u_list,
                w_l_list,
                b_l_list,
            ]
        else:
            raise ValueError(f"Unknown mode {mode}")
        if self.dc_decomp:
            output_list += [h_list, g_list]

        output = max_(output_list, axis=-1, dc_decomp=self.dc_decomp, mode=mode)

        return output

    def _pooling_function(self, inputs, pool_size, strides, padding, data_format, mode):

        if self.fast:
            return self._pooling_function_fast(inputs, pool_size, strides, padding, data_format, mode)
        else:
            return self._pooling_function_not_fast(inputs, pool_size, strides, padding, data_format, mode)

    def call(self, inputs):

        return self._pooling_function(inputs, self.pool_size, self.strides, self.padding, self.data_format, self.mode)


# Aliases
DecomonMaxPool2d = DecomonMaxPooling2D
