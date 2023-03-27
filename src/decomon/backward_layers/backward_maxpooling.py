from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Layer

from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import backward_max_
from decomon.layers.core import ForwardMode
from decomon.utils import Slope, get_lower, get_upper


class BackwardMaxPooling2D(BackwardLayer):
    """Backward  LiRPA of MaxPooling2D"""

    def __init__(
        self,
        layer: Layer,
        previous=True,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            previous=previous,
            **kwargs,
        )
        raise NotImplementedError()

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
        convex_domain=None,
    ):

        if convex_domain is None:
            convex_domain = {}
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
        convex_domain=None,
    ):

        if convex_domain is None:
            convex_domain = {}
        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:7]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:5]
            u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
            l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[:2]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        op_flat = Flatten()

        b_u_ = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")
        b_l_ = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")

        b_u_ = K.expand_dims(K.expand_dims(op_flat(b_u_), 1), -1)
        b_l_ = K.expand_dims(K.expand_dims(op_flat(b_l_), 1), -1)

        y = inputs[-1]
        n_out = w_out_u.shape[-1]

        w_out_u_ = K.concatenate([K.expand_dims(K.expand_dims(0 * (op_flat(y)), 1), -1)] * n_out, -1)
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
        convex_domain=None,
    ):
        """
        Args:
            inputs
            pool_size
            strides
            padding
            data_format

        Returns:

        """

        if convex_domain is None:
            convex_domain = {}
        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:7]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:5]
            u_c, l_c = 0, 0
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[:2]
            b_u, w_l, b_l, w_u = 0, 0, 0, 0
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        y = inputs[-1]

        input_shape = K.int_shape(y)

        if data_format in [None, "channels_last"]:
            axis = -1
        else:
            axis = 1

        # initialize vars
        x_0, u_c_list, w_u_list, b_u_list, l_c_list, w_l_list, b_l_list = None, None, None, None, None, None, None

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(u_c, input_shape[-1], -1)], -2)
            l_c_list = K.concatenate([self.internal_op(elem) for elem in tf.split(l_c, input_shape[-1], -1)], -2)

        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:

            b_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_u, input_shape[-1], -1)], -2)
            b_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(b_l, input_shape[-1], -1)], -2)
            w_u_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_u, input_shape[-1], -1)], -2)
            w_l_list = K.concatenate([self.internal_op(elem) for elem in tf.split(w_l, input_shape[-1], -1)], -2)

        if self.mode == ForwardMode.IBP:
            output_list = [
                u_c_list,
                l_c_list,
            ]
        elif self.mode == ForwardMode.HYBRID:
            output_list = [
                x_0,
                u_c_list,
                w_u_list,
                b_u_list,
                l_c_list,
                w_l_list,
                b_l_list,
            ]
        elif self.mode == ForwardMode.AFFINE:
            output_list = [
                x_0,
                u_c_list,
                w_u_list,
                b_u_list,
                l_c_list,
                w_l_list,
                b_l_list,
            ]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        w_out_u, b_out_u, w_out_l, b_out_l = backward_max_(
            output_list,
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=convex_domain,
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

    def call(self, inputs, **kwargs):

        x = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]
        return self._pooling_function(
            x, w_out_u, b_out_u, w_out_l, b_out_l, self.pool_size, self.strides, self.padding, self.data_format
        )
