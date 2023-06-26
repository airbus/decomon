from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Layer

from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import backward_max_
from decomon.core import ForwardMode, PerturbationDomain
from decomon.utils import get_lower, get_upper


class BackwardMaxPooling2D(BackwardLayer):
    """Backward  LiRPA of MaxPooling2D"""

    pool_size: Tuple[int, int]
    strides: Tuple[int, int]
    padding: str
    data_format: str
    fast: bool

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        raise NotImplementedError()

    def _pooling_function(
        self,
        inputs: List[tf.Tensor],
        w_u_out: tf.Tensor,
        b_u_out: tf.Tensor,
        w_l_out: tf.Tensor,
        b_l_out: tf.Tensor,
        pool_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: str,
        data_format: str,
        perturbation_domain: PerturbationDomain,
    ) -> List[tf.Tensor]:
        if self.fast:
            return self._pooling_function_fast(
                inputs=inputs,
                w_u_out=w_u_out,
                b_u_out=b_u_out,
                w_l_out=w_l_out,
                b_l_out=b_l_out,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                perturbation_domain=perturbation_domain,
            )
        else:
            return self._pooling_function_not_fast(
                inputs=inputs,
                w_u_out=w_u_out,
                b_u_out=b_u_out,
                w_l_out=w_l_out,
                b_l_out=b_l_out,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                perturbation_domain=perturbation_domain,
            )

    def _pooling_function_fast(
        self,
        inputs: List[tf.Tensor],
        w_u_out: tf.Tensor,
        b_u_out: tf.Tensor,
        w_l_out: tf.Tensor,
        b_l_out: tf.Tensor,
        pool_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: str,
        data_format: str,
        perturbation_domain: PerturbationDomain,
    ) -> List[tf.Tensor]:

        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:7]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:5]
            u_c = get_upper(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
            l_c = get_lower(x_0, w_l, b_l, perturbation_domain=perturbation_domain)
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[:2]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        op_flat = Flatten()

        b_u_pooled = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")
        b_l_pooled = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")

        b_u_pooled = K.expand_dims(K.expand_dims(op_flat(b_u_pooled), 1), -1)
        b_l_pooled = K.expand_dims(K.expand_dims(op_flat(b_l_pooled), 1), -1)

        y = inputs[-1]
        n_out = w_u_out.shape[-1]

        w_u_out_new = K.concatenate([K.expand_dims(K.expand_dims(0 * (op_flat(y)), 1), -1)] * n_out, -1)
        w_l_out_new = w_u_out_new

        b_u_out_new = (
            K.sum(K.maximum(w_u_out, 0) * b_u_pooled, 2) + K.sum(K.minimum(w_u_out, 0) * b_l_pooled, 2) + b_u_out
        )
        b_l_out_new = (
            K.sum(K.maximum(w_l_out, 0) * b_l_pooled, 2) + K.sum(K.minimum(w_l_out, 0) * b_u_pooled, 2) + b_l_out
        )

        return [w_u_out_new, b_u_out_new, w_l_out_new, b_l_out_new]

    def _pooling_function_not_fast(
        self,
        inputs: List[tf.Tensor],
        w_u_out: tf.Tensor,
        b_u_out: tf.Tensor,
        w_l_out: tf.Tensor,
        b_l_out: tf.Tensor,
        pool_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: str,
        data_format: str,
        perturbation_domain: PerturbationDomain,
    ) -> List[tf.Tensor]:
        """
        Args:
            inputs
            pool_size
            strides
            padding
            data_format

        Returns:

        """

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

        w_u_out, b_u_out, w_l_out, b_l_out = backward_max_(
            output_list,
            w_u_out,
            b_u_out,
            w_l_out,
            b_l_out,
            perturbation_domain=perturbation_domain,
            mode=self.mode,
            dc_decomp=self.dc_decomp,
            axis=-1,
        )

        # invert the convolution
        op_flat = Flatten()

        # do not do the activation so far
        # get input shape
        input_shape_channelreduced = list(inputs[0].shape[1:])
        n_axis = input_shape_channelreduced[axis]
        input_shape_channelreduced[axis] = 1
        n_dim = np.prod(input_shape_channelreduced)

        # create diagonal matrix
        id_list = [tf.linalg.diag(K.ones_like(op_flat(elem[0][None]))) for elem in tf.split(y, input_shape[axis], axis)]

        id_list = [K.reshape(identity_mat, [-1] + input_shape_channelreduced) for identity_mat in id_list]
        w_list = [self.internal_op(identity_mat) for identity_mat in id_list]

        # flatten
        weights = [K.reshape(op_flat(weights), (n_dim, -1, np.prod(pool_size))) for weights in w_list]

        n_0 = weights[0].shape[1]
        n_1 = weights[0].shape[2]

        w_u_out = K.reshape(w_u_out, (-1, 1, n_0, input_shape[axis], w_u_out.shape[-2], n_1))
        w_l_out = K.reshape(w_l_out, (-1, 1, n_0, input_shape[axis], w_l_out.shape[-2], n_1))

        weights = K.expand_dims(K.concatenate([K.expand_dims(K.expand_dims(w, -2), -2) for w in weights], 2), 0)

        w_u_out = K.reshape(
            K.sum(K.expand_dims(w_u_out, 1) * weights, (3, -1)), (-1, 1, n_dim * n_axis, w_u_out.shape[-2])
        )
        w_l_out = K.reshape(
            K.sum(K.expand_dims(w_l_out, 1) * weights, (3, -1)), (-1, 1, n_dim * n_axis, w_l_out.shape[-2])
        )

        return [w_u_out, b_u_out, w_l_out, b_l_out]

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        inputs_wo_backward_bounds = inputs[:-4]
        w_u_out, b_u_out, w_l_out, b_l_out = inputs[-4:]
        return self._pooling_function(
            inputs=inputs_wo_backward_bounds,
            w_u_out=w_u_out,
            b_u_out=b_u_out,
            w_l_out=w_l_out,
            b_l_out=b_l_out,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            perturbation_domain=self.perturbation_domain,
        )
