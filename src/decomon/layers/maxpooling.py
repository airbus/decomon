from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d
from tensorflow.keras.layers import InputSpec, MaxPooling2D

from decomon.core import PerturbationDomain
from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.utils import max_
from decomon.utils import get_lower, get_upper


# step 1: compute the maximum
class DecomonMaxPooling2D(DecomonLayer, MaxPooling2D):
    """LiRPA implementation of MaxPooling2D layers.
    See Keras official documentation for further details on the MaxPooling2D operator

    """

    original_keras_layer_class = MaxPooling2D

    pool_size: Tuple[int, int]
    strides: Tuple[int, int]
    padding: str
    data_format: str

    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int]] = (2, 2),
        strides: Optional[Union[int, Tuple[int, int]]] = None,
        padding: str = "valid",
        data_format: Optional[str] = None,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):

        super().__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
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
        self.filters = np.zeros((self.pool_size[0], self.pool_size[1], 1, np.prod(self.pool_size)), dtype=self.dtype)
        for i in range(self.pool_size[0]):
            for j in range(self.pool_size[1]):
                self.filters[i, j, 0, i * self.pool_size[0] + j] = 1

        def conv_(x: tf.Tensor) -> tf.Tensor:

            if self.data_format in [None, "channels_last"]:
                return K.cast(
                    K.expand_dims(
                        conv2d(
                            x,
                            self.filters,
                            strides=self.strides,
                            padding=self.padding,
                            data_format=self.data_format,
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
                            strides=self.strides,
                            padding=self.padding,
                            data_format=self.data_format,
                        ),
                        1,
                    ),
                    self.dtype,
                )

        self.internal_op = conv_

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        """
        Args:
            input_shape

        Returns:

        """

        if self.grad_bounds:
            raise NotImplementedError()

        output_shape_keras = MaxPooling2D.compute_output_shape(self, input_shape[0])
        input_dim = input_shape[-1][1]
        if self.mode == ForwardMode.IBP:
            output_shape = [output_shape_keras] * 2
        elif self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            x_shape = input_shape[1]
            w_shape = tuple([output_shape_keras[0], input_dim] + list(output_shape_keras)[1:])
            if self.mode == ForwardMode.AFFINE:
                output_shape = [x_shape] + [w_shape, output_shape_keras] * 2
            else:
                output_shape = [x_shape] + [output_shape_keras, w_shape, output_shape_keras] * 2
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output_shape += [output_shape_keras] * 2

        return output_shape

    def _pooling_function_fast(
        self,
        inputs: List[tf.Tensor],
        pool_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: str,
        data_format: str,
        mode: ForwardMode,
    ) -> List[tf.Tensor]:

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
            l_c_out = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")
            u_c_out = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")

        if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:

            if mode == ForwardMode.AFFINE:
                u_c = get_upper(x_0, w_u, b_u)
                l_c = get_lower(x_0, w_l, b_l)

                l_c_out = K.pool2d(l_c, pool_size, strides, padding, data_format, pool_mode="max")
                u_c_out = K.pool2d(u_c, pool_size, strides, padding, data_format, pool_mode="max")

            n_in = x_0.shape[-1]
            w_u_out = K.concatenate([0 * K.expand_dims(u_c_out, 1)] * n_in, 1)
            w_l_out = w_u_out
            b_u_out = u_c_out
            b_l_out = l_c_out

        if mode == ForwardMode.IBP:
            output = [u_c_out, l_c_out]
        elif mode == ForwardMode.HYBRID:
            output = [
                x_0,
                u_c_out,
                w_u_out,
                b_u_out,
                l_c_out,
                w_l_out,
                b_l_out,
            ]
        elif mode == ForwardMode.AFFINE:
            output = [
                x_0,
                w_u_out,
                b_u_out,
                w_l_out,
                b_l_out,
            ]
        else:
            raise ValueError(f"Unknown mode {mode}")
        return output

    def _pooling_function_not_fast(
        self,
        inputs: List[tf.Tensor],
        pool_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: str,
        data_format: str,
        mode: ForwardMode,
    ) -> List[tf.Tensor]:
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

    def _pooling_function(
        self,
        inputs: List[tf.Tensor],
        pool_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: str,
        data_format: str,
        mode: ForwardMode,
    ) -> List[tf.Tensor]:

        if self.fast:
            return self._pooling_function_fast(inputs, pool_size, strides, padding, data_format, mode)
        else:
            return self._pooling_function_not_fast(inputs, pool_size, strides, padding, data_format, mode)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        return self._pooling_function(inputs, self.pool_size, self.strides, self.padding, self.data_format, self.mode)


# Aliases
DecomonMaxPool2d = DecomonMaxPooling2D
