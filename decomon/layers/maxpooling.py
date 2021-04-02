from __future__ import absolute_import
from .core import DecomonLayer
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d
import numpy as np
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.keras.layers import MaxPooling2D
from decomon.layers.utils import max_
import tensorflow as tf


# step 1: compute the maximum
class DecomonMaxPooling2D(MaxPooling2D, DecomonLayer):
    def __init__(self, pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs):
        """
        Max pooling operation for 2D spatial data.
        see Keras official documentation

        :param pool_size: Integer or tuple of 2 integers, window size over which to take the maximum '(2,2)' will take
        the max value over a x2 pooling window. If only one integer is specified,
        the same window length will be used for both dimensions. Integer, tuple of 2 integers, or None.
        :param strides: Strides values. Specifies how far the pooling window moves for each pooling step.
        If None, it will default to `pool_size`.
        :param padding: One of `"valid"` or `"same"`(case - insensitive). `"valid"` means no padding.
        `"same"` results in padding evenly to the left / right or up / down of the input such that output
        has the same height / width dimension as the input.
        :param data_format: A string, one of `channels_last`(default) or `channels_first`.
        The ordering of the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first` corresponds to inputs
        with shape `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your Keras config file at `~ /.keras / keras.json`.
        If you never set it, then it will be "channels_last".
        :param kwargs:
        """

        super(DecomonMaxPooling2D, self).__init__(
            pool_size=pool_size, strides=strides, padding=padding, data_format=data_format, **kwargs
        )

        if self.grad_bounds:
            raise NotImplementedError()

        if self.dc_decomp:

            self.input_spec = [
                InputSpec(ndim=4),
                InputSpec(min_ndim=2),
                InputSpec(ndim=4),
                InputSpec(ndim=5),
                InputSpec(ndim=4),
                InputSpec(ndim=4),
                InputSpec(ndim=5),
                InputSpec(ndim=4),
                InputSpec(ndim=4),
                InputSpec(ndim=4),
            ]
        else:
            self.input_spec = [
                InputSpec(ndim=4),
                InputSpec(min_ndim=2),
                InputSpec(ndim=4),
                InputSpec(ndim=5),
                InputSpec(ndim=4),
                InputSpec(ndim=4),
                InputSpec(ndim=5),
                InputSpec(ndim=4),
            ]

        # express maxpooling with convolutions
        self.filters = np.zeros((pool_size[0], pool_size[1], 1, np.prod(pool_size)), dtype="float32")
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
        input_dim = input_shape[0][1]

        w_u_shape_, w_l_shape_ = [tuple([output_shape_[0], input_dim] + list(output_shape_)[1:])] * 2

        output_shape = (
            [output_shape_]
            + input_shape[2:3]
            + [output_shape_, w_u_shape_, output_shape_]
            + [output_shape_, w_l_shape_, output_shape_]
        )
        if self.dc_decomp:
            output_shape += [output_shape_, output_shape_]

        return output_shape

    def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
        """

        :param inputs:
        :param pool_size:
        :param strides:
        :param padding:
        :param data_format:
        :return:
        """
        if self.grad_bounds:
            raise NotImplementedError()

        if self.dc_decomp:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs

            # g_ = K.pool2d(g, pool_size, strides, padding, data_format,
            #              pool_mode='avg')
            # g_ = np.prod(pool_size) * g_

            # h_ = K.pool2d(h + g, pool_size, strides, padding, data_format,
            #                pool_mode='max') - g_

        else:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs

        y_ = K.pool2d(y, pool_size, strides, padding, data_format, pool_mode="max")
        input_shape = K.int_shape(y)

        if data_format in [None, "channels_last"]:
            axis = -1
        else:
            axis = 1

        y_list_ = [self.internal_op(elem) for elem in tf.split(y, input_shape[axis], axis)]
        y_list = K.concatenate(y_list_, -2)

        if self.dc_decomp:
            # h_list_ = [self.internal_op(elem) for elem in tf.split(h, input_shape[axis], axis)]
            # h_list = K.concatenate(h_list_, -2)
            h_list = K.concatenate(
                [self.internal_op(elem) for elem in tf.split(h, input_shape[-1], -1)],
                -2,
            )
            g_list = K.concatenate(
                [self.internal_op(elem) for elem in tf.split(g, input_shape[-1], -1)],
                -2,
            )
        b_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_u, input_shape[-1], -1)], -2)
        b_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_l, input_shape[-1], -1)], -2)
        u_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(u_c, input_shape[-1], -1)], -2)
        l_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(l_c, input_shape[-1], -1)], -2)

        w_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_u, input_shape[-1], -1)], -2)
        w_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_l, input_shape[-1], -1)], -2)

        output_list = [
            y_list,
            x_0,
            u_c_list,
            w_u_list,
            b_u_list,
            l_c_list,
            w_l_list,
            b_l_list,
        ]
        if self.dc_decomp:
            output_list += [h_list, g_list]

        output = max_(output_list, axis=-1, dc_decomp=self.dc_decomp)
        output[0] = y_

        return output

    def call(self, inputs):

        return self._pooling_function(inputs, self.pool_size, self.strides, self.padding, self.data_format)


# aliases
DecomonMaxPool2d = DecomonMaxPooling2D
