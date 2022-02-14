from __future__ import absolute_import
from .core import DecomonLayer
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d
import numpy as np

# from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.keras.layers import MaxPooling2D, InputSpec
from decomon.layers.utils import max_, get_lower, get_upper
import tensorflow as tf
from .core import F_FORWARD, F_IBP, F_HYBRID


# step 1: compute the maximum
class DecomonMaxPooling2D(MaxPooling2D, DecomonLayer):
    """
    LiRPA implementation of MaxPooling2D layers.
    See Keras official documentation for further details on the MaxPooling2D operator

    """

    def __init__(
        self, pool_size=(2, 2), strides=None, padding="valid", data_format=None, mode=F_HYBRID.name, fast=True, **kwargs
    ):

        super(DecomonMaxPooling2D, self).__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            mode=mode,
            **kwargs,
        )

        self.fast = fast

        if self.mode == F_IBP.name:
            self.input_spec = [
                # InputSpec(ndim=4),  # y
                # InputSpec(min_ndim=2),  # x
                InputSpec(ndim=4),  # u
                InputSpec(ndim=4),  # l
            ]

        if self.mode == F_FORWARD.name:
            self.input_spec = [
                # InputSpec(ndim=4),  # y
                InputSpec(min_ndim=2),  # x
                InputSpec(ndim=5),  # w_u
                InputSpec(ndim=4),  # b_u
                InputSpec(ndim=5),  # w_l
                InputSpec(ndim=4),
            ]  # b_l

        if self.mode == F_HYBRID.name:
            self.input_spec = [
                # InputSpec(ndim=4),  # y
                InputSpec(min_ndim=2),  # x
                InputSpec(ndim=4),  # u
                InputSpec(ndim=5),  # w_u
                InputSpec(ndim=4),  # b_u
                InputSpec(ndim=4),  # l
                InputSpec(ndim=5),  # w_l
                InputSpec(ndim=4),
            ]  # b_l

        if self.dc_decomp:
            self.input_spec += [InputSpec(ndim=4), InputSpec(ndim=4)]

        self.pool_size = pool_size

        # express maxpooling with convolutions
        if not self.fast or self.fast:
            self.filters = np.zeros((pool_size[0], pool_size[1], 1, np.prod(pool_size)), dtype=K.floatx())
            for i in range(pool_size[0]):
                for j in range(pool_size[1]):
                    self.filters[i, j, 0, i * pool_size[0] + j] = 1

            def conv_(x):

                if self.data_format in [None, "channels_last"]:
                    return K.expand_dims(
                        conv2d(
                            x,
                            self.filters,
                            strides=strides,
                            padding=padding,
                            data_format=data_format,
                        ),
                        -2,
                    )
                else:
                    return K.expand_dims(
                        conv2d(
                            x,
                            self.filters,
                            strides=strides,
                            padding=padding,
                            data_format=data_format,
                        ),
                        1,
                    )

            self.internal_op = conv_

    def compute_output_shape(self, input_shape):
        """

        :param input_shape:
        :return:
        """

        if self.grad_bounds:
            raise NotImplementedError()

        output_shape_ = super(DecomonMaxPooling2D, self).compute_output_shape(input_shape[0])
        input_dim = input_shape[-1][1]
        if self.mode in [F_HYBRID.name, F_FORWARD.name]:
            x_shape = input_shape[1]

        output_shape = []
        if self.mode == F_IBP.name:
            # output_shape = [output_shape_, input_shape[2:3], output_shape_, output_shape_]
            output_shape = [output_shape_] * 2
        if self.mode in [F_HYBRID.name, F_FORWARD.name]:

            w_shape_ = tuple([output_shape_[0], input_dim] + list(output_shape_)[1:])
            if self.mode == F_FORWARD.name:
                # output_shape = [output_shape_, input_shape[2:3], w_shape_, output_shape_, w_shape_, output_shape_]
                output_shape = [x_shape] + [w_shape_, output_shape_] * 2
            if self.mode == F_HYBRID.name:
                # output_shape = [output_shape_,input_shape[2:3],output_shape_,w_shape_,output_shape_,output_shape_,
                #    w_shape_,
                #    output_shape_,
                # ]
                output_shape = [x_shape] + [output_shape_, w_shape_, output_shape_] * 2

        if self.dc_decomp:
            output_shape += [output_shape_] * 2

        return output_shape

    def _pooling_function_fast(self, inputs, pool_size, strides, padding, data_format, mode):

        if self.dc_decomp:
            raise NotImplementedError()

        if mode == F_HYBRID.name:
            # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]

        if mode == F_FORWARD.name:
            # y, x_0, w_u, b_u, w_l, b_l = inputs[:6]
            x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]
        if mode == F_IBP.name:
            # y, x_0, u_c, l_c = inputs[:4]
            u_c, l_c = inputs[: self.nb_tensors]

        # y_ = K.pool2d(y, pool_size, strides, padding, data_format, pool_mode="max")

        if mode in [F_IBP.name, F_HYBRID.name]:
            l_c_ = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")
            u_c_ = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")

        if mode in [F_FORWARD.name, F_HYBRID.name]:

            if mode in F_FORWARD.name:
                u_c = get_upper(x_0, w_u, b_u)
                l_c = get_lower(x_0, w_l, b_l)

                l_c_ = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")
                u_c_ = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")

            n_in = x_0.shape[-1]
            w_u_ = K.concatenate([0 * K.expand_dims(u_c_, 1)] * n_in, 1)
            w_l_ = w_u_
            b_u_ = u_c_
            b_l_ = l_c_

        if mode == F_IBP.name:
            # output = [y_,x_0,u_c_,l_c_,]
            output = [u_c_, l_c_]
        if mode == F_HYBRID.name:
            # output = [y_,x_0,u_c_,w_u_,b_u_,l_c_,w_l_,b_l_,]
            output = [
                x_0,
                u_c_,
                w_u_,
                b_u_,
                l_c_,
                w_l_,
                b_l_,
            ]
        if mode == F_FORWARD.name:
            # output = [y_,x_0,w_u_,b_u_,w_l_,b_l_,]
            output = [
                x_0,
                w_u_,
                b_u_,
                w_l_,
                b_l_,
            ]

        return output

    def _pooling_function_not_fast(self, inputs, pool_size, strides, padding, data_format, mode):
        nb_tensors = self.nb_tensors
        if self.dc_decomp:
            h, g = inputs[-2:]
            nb_tensors -= 2

        if mode == F_HYBRID.name:
            # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        if mode == F_FORWARD.name:
            # y, x_0, w_u, b_u, w_l, b_l = inputs[:6]
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
        if mode == F_IBP.name:
            # y, x_0, u_c, l_c = inputs[:4]
            u_c, l_c = inputs[:nb_tensors]

        # y_ = K.pool2d(inputs[-1], pool_size, strides, padding, data_format, pool_mode="max")
        input_shape = K.int_shape(inputs[-1])

        if data_format in [None, "channels_last"]:
            axis = -1
        else:
            axis = 1

        # y_list_ = [self.internal_op(elem) for elem in tf.split(y, input_shape[axis], axis)]
        # y_list = K.concatenate(y_list_, -2)

        if self.dc_decomp:
            h_list = K.concatenate(
                [self.internal_op(elem) for elem in tf.split(h, input_shape[-1], -1)],
                -2,
            )
            g_list = K.concatenate(
                [self.internal_op(elem) for elem in tf.split(g, input_shape[-1], -1)],
                -2,
            )

        if mode in [F_IBP.name, F_HYBRID.name]:
            u_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(u_c, input_shape[-1], -1)], -2)
            l_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(l_c, input_shape[-1], -1)], -2)

        if mode in [F_FORWARD.name, F_HYBRID.name]:

            b_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_u, input_shape[-1], -1)], -2)
            b_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_l, input_shape[-1], -1)], -2)
            w_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_u, input_shape[-1], -1)], -2)
            w_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_l, input_shape[-1], -1)], -2)

        if mode == F_IBP.name:
            # output_list = [y_list,x_0,u_c_list,l_c_list,]
            output_list = [u_c_list, l_c_list]
        if mode == F_HYBRID.name:
            # output_list = [y_list,x_0,u_c_list,w_u_list,b_u_list,
            #    l_c_list,w_l_list,b_l_list,]
            output_list = [
                x_0,
                u_c_list,
                w_u_list,
                b_u_list,
                l_c_list,
                w_l_list,
                b_l_list,
            ]
        if mode == F_FORWARD.name:
            # output_list = [y_list,x_0,w_u_list,b_u_list,w_l_list,b_l_list,]
            output_list = [
                x_0,
                w_u_list,
                b_u_list,
                w_l_list,
                b_l_list,
            ]

        if self.dc_decomp:
            output_list += [h_list, g_list]

        output = max_(output_list, axis=-1, dc_decomp=self.dc_decomp, mode=mode)
        # output[0] = y_

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
