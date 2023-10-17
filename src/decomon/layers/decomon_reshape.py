from typing import Any, Dict, List, Optional, Tuple, Type, Union

import keras_core as keras
import keras_core.ops as K
from keras_core.layers import InputSpec, Layer, Permute, Reshape
from keras_core.src.backend import rnn

from decomon.core import ForwardMode, PerturbationDomain, get_affine, get_ibp
from decomon.layers.core import DecomonLayer


class DecomonReshape(DecomonLayer, Reshape):
    """Forward LiRPA implementation of Reshape layers.
    See Keras official documentation for further details on the Reshape operator
    """

    original_keras_layer_class = Reshape

    def __init__(
        self,
        target_shape: Tuple[int, ...],
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            data_format
            **kwargs
        """
        super().__init__(
            target_shape=target_shape,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

        if self.mode == ForwardMode.HYBRID:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # l_c
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]
        elif self.mode == ForwardMode.IBP:
            self.input_spec = [
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # l_c
            ]
        if self.mode == ForwardMode.AFFINE:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]

        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=1), InputSpec(min_ndim=1)]

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        def op(x: keras.KerasTensor) -> keras.KerasTensor:
            return Reshape.call(self, x)

        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=False
        )
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        if self.dc_decomp:
            h_out = op(h)
            g_out = op(g)
        else:
            h_out, g_out = empty_tensor, empty_tensor

        if self.ibp:
            u_c_out = op(u_c)
            l_c_out = op(l_c)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            b_u_out = op(b_u)
            b_l_out = op(b_l)

            if len(w_u.shape) == len(b_u.shape):
                w_u_out = op(w_u)
                w_l_out = op(w_l)

            else:

                def step_func(
                    x: keras.KerasTensor, _: List[keras.KerasTensor]
                ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
                    return op(x), _

                w_u_out = rnn(step_function=step_func, inputs=w_u, initial_states=[], unroll=False)[1]
                w_l_out = rnn(step_function=step_func, inputs=w_l, initial_states=[], unroll=False)[1]
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )


class DecomonPermute(DecomonLayer, Permute):
    """Forward LiRPA implementation of Reshape layers.
    See Keras official documentation for further details on the Reshape operator
    """

    original_keras_layer_class = Permute

    def __init__(
        self,
        dims: Tuple[int, ...],
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            data_format
            **kwargs
        """
        super().__init__(
            dims=dims,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

        if self.mode == ForwardMode.HYBRID:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # l_c
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]
        elif self.mode == ForwardMode.IBP:
            self.input_spec = [
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # l_c
            ]
        elif self.mode == ForwardMode.AFFINE:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]
        else:
            raise ValueError(f"Unknown mode {mode}")

        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=1), InputSpec(min_ndim=1)]

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        def op(x: keras.KerasTensor) -> keras.KerasTensor:
            return Permute.call(self, x)

        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=False
        )
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        if self.dc_decomp:
            h_out = op(h)
            g_out = op(g)
        else:
            h_out, g_out = empty_tensor, empty_tensor

        if self.ibp:
            u_c_out = op(u_c)
            l_c_out = op(l_c)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            b_u_out = op(b_u)
            b_l_out = op(b_l)

            if len(w_u.shape) == len(b_u.shape):
                w_u_out = op(w_u)
                w_l_out = op(w_l)
            else:

                def step_func(
                    x: keras.KerasTensor, _: List[keras.KerasTensor]
                ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
                    return op(x), _

                w_u_out = rnn(step_function=step_func, inputs=w_u, initial_states=[], unroll=False)[1]
                w_l_out = rnn(step_function=step_func, inputs=w_l, initial_states=[], unroll=False)[1]
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )
