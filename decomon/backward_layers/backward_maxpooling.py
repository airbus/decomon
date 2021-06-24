from __future__ import absolute_import
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, MaxPooling2D, Flatten
from decomon.layers.maxpooling import DecomonMaxPooling2D, DecomonMaxPool2d
from .utils import V_slope, backward_max_
from tensorflow.keras.backend import conv2d
from ..layers.core import F_FORWARD, F_IBP, F_HYBRID
from decomon.layers.utils import get_lower, get_upper


class BackwardMaxPooling2D(Layer):
    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardMaxPooling2D, self).__init__(**kwargs)
        if not isinstance(layer, MaxPooling2D):
            raise NotImplementedError()
        self.mode = layer.mode
        self.pool_size = layer.pool_size
        self.strides = layer.strides
        self.padding = layer.padding
        self.data_format = layer.data_format

        # express maxpooling with convolutions
        self.filters = layer.filters
        self.grad_bounds = layer.grad_bounds
        self.internal_op = layer.internal_op
        self.mode = layer.mode
        self.fast = layer.fast

    def _pooling_function(
        self,
        inputs,
        w_out_u,
        b_out_u,
        w_out_l,
        b_out_l,
        pool_size,
        strides,
        padding,
        data_format,
        convex_domain={},
        slope=V_slope.name,
    ):

        if self.fast:
            return self._pooling_function_fast(
                inputs,
                w_out_u,
                b_out_u,
                w_out_l,
                b_out_l,
                pool_size,
                strides,
                padding,
                data_format,
                convex_domain,
                slope,
            )
        else:
            return self._pooling_function_not_fast(
                inputs,
                w_out_u,
                b_out_u,
                w_out_l,
                b_out_l,
                pool_size,
                strides,
                padding,
                data_format,
                convex_domain,
                slope,
            )

    def _pooling_function_fast(
        self,
        inputs,
        w_out_u,
        b_out_u,
        w_out_l,
        b_out_l,
        pool_size,
        strides,
        padding,
        data_format,
        convex_domain={},
        slope=V_slope.name,
    ):
        if self.grad_bounds:
            raise NotImplementedError()

        if self.mode == F_HYBRID.name:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]
        if self.mode == F_FORWARD.name:
            y, x_0, w_u, b_u, w_l, b_l = inputs[:6]
        if self.mode == F_IBP.name:
            y, x_0, u_c, l_c = inputs[:4]

        if self.mode == F_FORWARD.name:

            u_c = get_upper(x_0, w_u, b_u)
            l_c = get_lower(x_0, w_l, b_l)
        op_flat = Flatten()

        b_u_ = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")
        b_l_ = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")

        b_u_ = K.expand_dims(K.expand_dims(op_flat(b_u_), 1), -1)
        b_l_ = K.expand_dims(K.expand_dims(op_flat(b_l_), 1), -1)

        n_in = np.prod(y.shape[1:])
        n_out = w_out_u.shape[-1]

        w_out_u_ = K.concatenate([K.expand_dims(K.expand_dims(0 * (op_flat(y)), 1), -1)] * n_out, -1)
        # w_out_u_ = K.expand_dims(K.zeros((1, n_in, n_out)), 1) +  K.expand_dims(K.expand_dims(K.zeros_like(op_flat(y)), 1), -1)
        w_out_l_ = w_out_u_

        b_out_u_ = K.sum(K.maximum(w_out_u, 0) * b_u_, 2) + K.sum(K.minimum(w_out_u, 0) * b_l_, 2) + b_out_u
        b_out_l_ = K.sum(K.maximum(w_out_l, 0) * b_l_, 2) + K.sum(K.minimum(w_out_l, 0) * b_u_, 2) + b_out_l

        return w_out_u_, b_out_u_, w_out_l_, b_out_l_

    def _pooling_function_not_fast(
        self,
        inputs,
        w_out_u,
        b_out_u,
        w_out_l,
        b_out_l,
        pool_size,
        strides,
        padding,
        data_format,
        convex_domain={},
        slope=V_slope.name,
    ):
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

        if self.mode == F_HYBRID.name:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]
        if self.mode == F_FORWARD.name:
            y, x_0, w_u, b_u, w_l, b_l = inputs[:6]
        if self.mode == F_IBP.name:
            y, x_0, u_c, l_c = inputs[:4]

        input_shape = K.int_shape(y)

        if data_format in [None, "channels_last"]:
            axis = -1
        else:
            axis = 1

        y_list_ = [self.internal_op(elem) for elem in tf.split(y, input_shape[axis], axis)]
        y_list = K.concatenate(y_list_, -2)

        if self.mode in [F_IBP.name, F_HYBRID.name]:
            u_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(u_c, input_shape[-1], -1)], -2)
            l_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(l_c, input_shape[-1], -1)], -2)

        if self.mode in [F_FORWARD.name, F_HYBRID.name]:

            b_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_u, input_shape[-1], -1)], -2)
            b_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_l, input_shape[-1], -1)], -2)
            w_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_u, input_shape[-1], -1)], -2)
            w_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_l, input_shape[-1], -1)], -2)

        if self.mode == F_IBP.name:
            output_list = [
                y_list,
                x_0,
                u_c_list,
                l_c_list,
            ]
        if self.mode == F_HYBRID.name:
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
        if self.mode == F_FORWARD.name:
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

        w_out_u, b_out_u, w_out_l, b_out_l = backward_max_(
            output_list,
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=convex_domain,
            slope=slope,
            mode=self.mode,
            axis=-1,
        )

        # invert the convolution
        op_flat = Flatten()

        # do not do the activation so far
        # get input shape
        input_shape_ = list(inputs[0].shape[1:])
        n_axis = input_shape_[axis]
        input_shape_[axis] = 1
        n_dim = np.prod(input_shape_)

        # import pdb; pdb.set_trace()
        # create diagonal matrix

        id_list_ = [
            tf.linalg.diag(K.ones_like(op_flat(elem[0][None]))) for elem in tf.split(y, input_shape[axis], axis)
        ]

        id_list = [K.reshape(id_, [-1] + input_shape_) for id_ in id_list_]
        w_list = [self.internal_op(id_) for id_ in id_list]

        # flatten
        weights = [K.reshape(op_flat(weights), (n_dim, -1, np.prod(pool_size))) for weights in w_list]

        n_0 = weights[0].shape[1]
        n_1 = weights[0].shape[2]

        w_out_u = K.reshape(w_out_u, (-1, 1, n_0, input_shape[axis], w_out_u.shape[-2], n_1))
        w_out_l = K.reshape(w_out_l, (-1, 1, n_0, input_shape[axis], w_out_l.shape[-2], n_1))

        weights = K.expand_dims(K.concatenate([K.expand_dims(K.expand_dims(w, -2), -2) for w in weights], 2), 0)

        w_out_u_ = K.reshape(
            K.sum(K.expand_dims(w_out_u, 1) * weights, (3, -1)), (-1, 1, n_dim * n_axis, w_out_u.shape[-2])
        )
        w_out_l_ = K.reshape(
            K.sum(K.expand_dims(w_out_l, 1) * weights, (3, -1)), (-1, 1, n_dim * n_axis, w_out_l.shape[-2])
        )

        return w_out_u_, b_out_u, w_out_l_, b_out_l

    def call(self, inputs):

        x = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]
        return self._pooling_function(
            x, w_out_u, b_out_u, w_out_l, b_out_l, self.pool_size, self.strides, self.padding, self.data_format
        )
