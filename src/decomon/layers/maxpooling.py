from typing import Any, Dict, List, Optional, Tuple, Union

import keras.ops as K
import numpy as np
from keras.layers import InputSpec, MaxPooling2D

from decomon.core import ForwardMode, PerturbationDomain
from decomon.layers.core import DecomonLayer
from decomon.layers.utils import max_
from decomon.types import BackendTensor


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
        self.filters = np.zeros(
            (self.pool_size[0], self.pool_size[1], 1, int(np.prod(self.pool_size))), dtype=self.dtype
        )
        for i in range(self.pool_size[0]):
            for j in range(self.pool_size[1]):
                self.filters[i, j, 0, i * self.pool_size[0] + j] = 1

        self.filters_w = self.filters[None]
        self.strides_w = (1,) + self.strides

        def conv_(x: BackendTensor) -> BackendTensor:
            if self.data_format in [None, "channels_last"]:
                return K.cast(
                    K.expand_dims(
                        K.conv(
                            x,
                            self.filters,
                            strides=list(self.strides),
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
                        K.conv(
                            x,
                            self.filters,
                            strides=list(self.strides),
                            padding=self.padding,
                            data_format=self.data_format,
                        ),
                        1,
                    ),
                    self.dtype,
                )

        def conv_w_(x: BackendTensor) -> BackendTensor:
            if self.data_format in [None, "channels_last"]:
                return K.cast(
                    K.expand_dims(
                        K.conv(
                            x,
                            self.filters_w,
                            strides=list(self.strides_w),
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
                        K.conv(
                            x,
                            self.filters_w,
                            strides=list(self.strides_w),
                            padding=self.padding,
                            data_format=self.data_format,
                        ),
                        1,
                    ),
                    self.dtype,
                )

        self.internal_op = conv_
        self.internal_op_w = conv_w_

    def _pooling_function_fast(
        self,
        inputs: List[BackendTensor],
    ) -> List[BackendTensor]:
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        l_c_out = K.max_pool(l_c, self.pool_size, self.strides, self.padding, self.data_format)
        u_c_out = K.max_pool(u_c, self.pool_size, self.strides, self.padding, self.data_format)

        if self.affine:
            n_in = x.shape[-1]
            w_u_out = K.concatenate([0 * K.expand_dims(u_c_out, 1)] * n_in, 1)
            w_l_out = w_u_out
            b_u_out = u_c_out
            b_l_out = l_c_out
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        if self.dc_decomp:
            raise NotImplementedError()
        else:
            h_out, g_out = empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )

    def _pooling_function_not_fast(
        self,
        inputs: List[BackendTensor],
    ) -> List[BackendTensor]:
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=False
        )
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)
        input_shape = self.inputs_outputs_spec.get_kerasinputshape(inputs)
        n_split = input_shape[-1]

        if self.dc_decomp:
            h_out = K.concatenate(
                [self.internal_op(elem) for elem in K.split(h, n_split, -1)],
                -2,
            )
            g_out = K.concatenate(
                [self.internal_op(elem) for elem in K.split(g, n_split, -1)],
                -2,
            )
        else:
            h_out, g_out = empty_tensor, empty_tensor

        if self.ibp:
            u_c_out = K.concatenate([self.internal_op(elem) for elem in K.split(u_c, n_split, -1)], -2)
            l_c_out = K.concatenate([self.internal_op(elem) for elem in K.split(l_c, n_split, -1)], -2)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            b_u_out = K.concatenate([self.internal_op(elem) for elem in K.split(b_u, n_split, -1)], -2)
            b_l_out = K.concatenate([self.internal_op(elem) for elem in K.split(b_l, n_split, -1)], -2)
            w_u_out = K.concatenate([self.internal_op_w(elem) for elem in K.split(w_u, n_split, -1)], -2)
            w_l_out = K.concatenate([self.internal_op_w(elem) for elem in K.split(w_l, n_split, -1)], -2)
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        outputs = self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )

        return max_(
            outputs, axis=-1, dc_decomp=self.dc_decomp, mode=self.mode, perturbation_domain=self.perturbation_domain
        )

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
        if self.fast:
            return self._pooling_function_fast(inputs)
        else:
            return self._pooling_function_not_fast(inputs)


# Aliases
DecomonMaxPool2d = DecomonMaxPooling2D
