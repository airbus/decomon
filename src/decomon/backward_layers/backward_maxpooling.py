from typing import Any, Dict, List, Optional, Tuple, Union

import keras_core as keras
import keras_core.ops as K
import numpy as np
import tensorflow as tf
from keras_core.layers import Flatten, Layer

from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import backward_max_, get_identity_lirpa
from decomon.core import ForwardMode, PerturbationDomain


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

    def _pooling_function_fast(
        self,
        inputs: List[keras.KerasTensor],
        w_u_out: keras.KerasTensor,
        b_u_out: keras.KerasTensor,
        w_l_out: keras.KerasTensor,
        b_l_out: keras.KerasTensor,
    ) -> List[keras.KerasTensor]:
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)

        op_flat = Flatten()

        b_u_pooled = K.max_pool(u_c, self.pool_size, self.strides, self.padding, self.data_format)
        b_l_pooled = K.max_pool(l_c, self.pool_size, self.strides, self.padding, self.data_format)

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
        inputs: List[keras.KerasTensor],
        w_u_out: keras.KerasTensor,
        b_u_out: keras.KerasTensor,
        w_l_out: keras.KerasTensor,
        b_l_out: keras.KerasTensor,
    ) -> List[keras.KerasTensor]:
        """
        Args:
            inputs
            pool_size
            strides
            padding
            data_format

        Returns:

        """
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)
        y = inputs[-1]
        input_shape = y.shape

        if self.data_format in [None, "channels_last"]:
            axis = -1
        else:
            axis = 1

        # initialize vars
        u_c_tmp, w_u_tmp, b_u_tmp, l_c_tmp, w_l_tmp, b_l_tmp = (
            empty_tensor,
            empty_tensor,
            empty_tensor,
            empty_tensor,
            empty_tensor,
            empty_tensor,
        )

        if self.ibp:
            u_c_tmp = K.concatenate([self.internal_op(elem) for elem in K.split(u_c, input_shape[-1], -1)], -2)
            l_c_tmp = K.concatenate([self.internal_op(elem) for elem in K.split(l_c, input_shape[-1], -1)], -2)

        if self.affine:
            b_u_tmp = K.concatenate([self.internal_op(elem) for elem in K.split(b_u, input_shape[-1], -1)], -2)
            b_l_tmp = K.concatenate([self.internal_op(elem) for elem in K.split(b_l, input_shape[-1], -1)], -2)
            w_u_tmp = K.concatenate([self.internal_op(elem) for elem in K.split(w_u, input_shape[-1], -1)], -2)
            w_l_tmp = K.concatenate([self.internal_op(elem) for elem in K.split(w_l, input_shape[-1], -1)], -2)

        if self.dc_decomp:
            raise NotImplementedError()
        else:
            h_tmp, g_tmp = empty_tensor, empty_tensor

        outputs_tmp = self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_tmp, w_u_tmp, b_u_tmp, l_c_tmp, w_l_tmp, b_l_tmp, h_tmp, g_tmp]
        )

        w_u_out, b_u_out, w_l_out, b_l_out = backward_max_(
            outputs_tmp,
            w_u_out,
            b_u_out,
            w_l_out,
            b_l_out,
            perturbation_domain=self.perturbation_domain,
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
        n_dim = int(np.prod(input_shape_channelreduced))

        # create diagonal matrix
        id_list = [tf.linalg.diag(K.ones_like(op_flat(elem[0][None]))) for elem in K.split(y, input_shape[axis], axis)]

        id_list = [K.reshape(identity_mat, [-1] + input_shape_channelreduced) for identity_mat in id_list]
        w_list = [self.internal_op(identity_mat) for identity_mat in id_list]

        # flatten
        weights = [K.reshape(op_flat(weights), (n_dim, -1, int(np.prod(self.pool_size)))) for weights in w_list]

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

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)
        if self.fast:
            return self._pooling_function_fast(
                inputs=inputs,
                w_u_out=w_u_out,
                b_u_out=b_u_out,
                w_l_out=w_l_out,
                b_l_out=b_l_out,
            )
        else:
            return self._pooling_function_not_fast(
                inputs=inputs,
                w_u_out=w_u_out,
                b_u_out=b_u_out,
                w_l_out=w_l_out,
                b_l_out=b_l_out,
            )
