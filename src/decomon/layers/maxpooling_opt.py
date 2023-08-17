from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d
from tensorflow.keras.layers import InputSpec, MaxPooling2D

from decomon.core import PerturbationDomain
from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.utils import max_
from decomon.layers.utils_pooling import get_upper_linear_hull_max, get_lower_linear_hull_max, get_conv_op_config
#from decomon.utils import get_lower, get_upper


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

        self.kernel = None
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            kernel = get_conv_op_config(self.get_config())
            self.kernel = kernel

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

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]
            u_c = self.perturbation_domain.get_upper(x_0, w_u, b_u)
            l_c = self.perturbation_domain.get_lower(x_0, w_l, b_l)
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[: self.nb_tensors]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            l_c_out = K.pool2d(l_c, self.pool_size, self.strides, self.padding, self.data_format, pool_mode="max")
            u_c_out = K.pool2d(u_c, self.pool_size, self.strides, self.padding, self.data_format, pool_mode="max")
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            
            # apply convolution per channel
            if self.data_format=='channel_first':
                raise NotImplementedError()
            else:
                n_channel = inputs[-1].shape[-1]
                w_u_split = tf.split(w_u, n_channel, -1)
                b_u_split = tf.split(b_u, n_channel, -1)
                w_l_split = tf.split(w_l, n_channel, -1)
                b_l_split = tf.split(b_l, n_channel, -1)

                u_split = tf.split(u_c, n_channel, -1)
                l_split = tf.split(l_c, n_channel, -1)

            def conv_pool(x: tf.Tensor) -> tf.Tensor:
                strides_ = list(self.strides)
                if len(strides_)==2 and strides_[0]!=strides_[1]:
                    raise NotImplementedError("different values of strides for MaxPooling are not supported but received {}", self.strides)
                return K.conv2d(
                    x,
                    self.kernel,
                    strides=list(self.strides)[0],
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=1
            )
            w_u_conv = K.concatenate([K.expand_dims(conv_pool(e), -2) for e in w_u_split], -2)
            w_l_conv = K.concatenate([K.expand_dims(conv_pool(e), -2) for e in w_l_split], -2)
            b_u_conv = K.concatenate([K.expand_dims(conv_pool(e), -2) for e in b_u_split], -2)
            b_l_conv = K.concatenate([K.expand_dims(conv_pool(e), -2) for e in b_l_split], -2)

            u_conv = K.concatenate([K.expand_dims(conv_pool(e), -2) for e in u_split], -2)
            l_conv = K.concatenate([K.expand_dims(conv_pool(e), -2) for e in l_split], -2)

            w_u_max, b_u_max = get_upper_linear_hull_max([u_conv, l_conv], axis=-1,mode=ForwardMode.IBP, perturbation_domain=self.perturbation_domain)
            w_l_max, b_l_max = get_lower_linear_hull_max([u_conv, l_conv], axis=-1,
                                                        mode=ForwardMode.IBP, perturbation_domain=self.perturbation_domain)
        
            # compute bias
            b_u_out = b_u_max + K.sum(w_u_max*b_u_conv, -1)
            b_l_out = b_l_max + K.sum(w_l_max*b_l_conv, -1)

            if len(w_u_conv.shape)>len(w_u_max.shape):
                w_u_max = K.expand_dims(w_u_max, 1)
                w_l_max = K.expand_dims(w_l_max, 1)
            w_u_out = K.sum(w_u_max*w_u_conv, -1)
            w_l_out = K.sum(w_l_max*w_l_conv, -1)
            
        if self.mode == ForwardMode.IBP:
            output = [u_c_out, l_c_out]
        elif self.mode == ForwardMode.HYBRID:
            output = [
                x_0,
                u_c_out,
                w_u_out,
                b_u_out,
                l_c_out,
                w_l_out,
                b_l_out,
            ]
        elif self.mode == ForwardMode.AFFINE:
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


# Aliases
DecomonMaxPool2d = DecomonMaxPooling2D
