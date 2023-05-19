from typing import Any, Dict, List, Optional, Tuple, Type, Union

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Layer, Permute, Reshape

from decomon.layers.core import DecomonLayer, ForwardMode


class DecomonReshape(DecomonLayer, Reshape):
    """Forward LiRPA implementation of Reshape layers.
    See Keras official documentation for further details on the Reshape operator
    """

    original_keras_layer_class = Reshape

    def __init__(
        self,
        target_shape: Tuple[int, ...],
        convex_domain: Optional[Dict[str, Any]] = None,
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
            convex_domain=convex_domain,
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

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        def op(x: tf.Tensor) -> tf.Tensor:
            return Reshape.call(self, x)

        nb_tensors = self.nb_tensors
        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = op(h)
            g_ = op(g)
            nb_tensors -= 2

        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[:nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_ = op(u_c)
            l_c_ = op(l_c)

        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            b_u_ = op(b_u)
            b_l_ = op(b_l)

            if len(w_u.shape) == len(b_u.shape):
                w_u_ = op(w_u)
                w_l_ = op(w_l)

            else:

                def step_func(x: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
                    return op(x), _

                w_u_ = K.rnn(step_function=step_func, inputs=w_u, initial_states=[], unroll=False)[1]
                w_l_ = K.rnn(step_function=step_func, inputs=w_l, initial_states=[], unroll=False)[1]

        if self.mode == ForwardMode.HYBRID:
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        elif self.mode == ForwardMode.AFFINE:
            output = [x_0, w_u_, b_u_, w_l_, b_l_]
        elif self.mode == ForwardMode.IBP:
            output = [u_c_, l_c_]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output += [h_, g_]

        return output


class DecomonPermute(DecomonLayer, Permute):
    """Forward LiRPA implementation of Reshape layers.
    See Keras official documentation for further details on the Reshape operator
    """

    original_keras_layer_class = Permute

    def __init__(
        self,
        dims: Tuple[int, ...],
        convex_domain: Optional[Dict[str, Any]] = None,
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
            convex_domain=convex_domain,
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

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        def op(x: tf.Tensor) -> tf.Tensor:
            return Permute.call(self, x)

        nb_tensors = self.nb_tensors
        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = op(h)
            g_ = op(g)
            nb_tensors -= 2

        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[:nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_ = op(u_c)
            l_c_ = op(l_c)

        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            b_u_ = op(b_u)
            b_l_ = op(b_l)

            if len(w_u.shape) == len(b_u.shape):
                w_u_ = op(w_u)
                w_l_ = op(w_l)
            else:

                def step_func(x: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
                    return op(x), _

                w_u_ = K.rnn(step_function=step_func, inputs=w_u, initial_states=[], unroll=False)[1]
                w_l_ = K.rnn(step_function=step_func, inputs=w_l, initial_states=[], unroll=False)[1]

        if self.mode == ForwardMode.HYBRID:
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        elif self.mode == ForwardMode.AFFINE:
            output = [x_0, w_u_, b_u_, w_l_, b_l_]
        elif self.mode == ForwardMode.IBP:
            output = [u_c_, l_c_]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output += [h_, g_]

        return output
