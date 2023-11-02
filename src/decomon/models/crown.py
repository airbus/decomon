# extra layers necessary for backward LiRPA
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import keras.ops as K
import tensorflow as tf
from keras.layers import InputSpec, Layer
from keras.src.layers.merging.dot import batch_dot

from decomon.core import ForwardMode, PerturbationDomain
from decomon.keras_utils import BatchedDiagLike


class Fuse(Layer):
    def __init__(self, mode: Union[str, ForwardMode], **kwargs: Any):
        super().__init__(**kwargs)
        self.mode = ForwardMode(mode)

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        inputs_wo_backward_bounds = inputs[:-4]
        backward_bounds = inputs[-4:]

        if self.mode == ForwardMode.AFFINE:
            x_0, w_f_u, b_f_u, w_f_l, b_f_l = inputs_wo_backward_bounds
        elif self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_f_u, b_f_u, l_c, w_f_l, b_f_l = inputs_wo_backward_bounds
        else:
            return backward_bounds

        return merge_with_previous([w_f_u, b_f_u, w_f_l, b_f_l] + backward_bounds)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"mode": self.mode})
        return config


class Convert2BackwardMode(Layer):
    def __init__(self, mode: Union[str, ForwardMode], perturbation_domain: PerturbationDomain, **kwargs: Any):
        super().__init__(**kwargs)
        self.mode = ForwardMode(mode)
        self.perturbation_domain = perturbation_domain

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        inputs_wo_backward_bounds = inputs[:-4]
        backward_bounds = inputs[-4:]
        w_u_out, b_u_out, w_l_out, b_l_out = backward_bounds
        empty_tensor = K.convert_to_tensor([])

        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            x_0 = inputs_wo_backward_bounds[0]
        else:
            u_c, l_c = inputs_wo_backward_bounds
            x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_out = self.perturbation_domain.get_upper(x_0, w_u_out, b_u_out)
            l_c_out = self.perturbation_domain.get_lower(x_0, w_l_out, b_l_out)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.mode == ForwardMode.AFFINE:
            return [x_0] + backward_bounds
        elif self.mode == ForwardMode.IBP:
            return [u_c_out, l_c_out]
        elif self.mode == ForwardMode.HYBRID:
            return [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]

        else:
            raise ValueError(f"Unknwon mode {self.mode}")

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"mode": self.mode, "perturbation_domain": self.perturbation_domain})
        return config


class MergeWithPrevious(Layer):
    def __init__(
        self,
        input_shape_layer: Optional[Tuple[int, ...]] = None,
        backward_shape_layer: Optional[Tuple[int, ...]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.input_shape_layer = input_shape_layer
        self.backward_shape_layer = backward_shape_layer
        if not (input_shape_layer is None) and not (backward_shape_layer is None):
            _, n_in, n_h = input_shape_layer
            _, n_h, n_out = backward_shape_layer

            w_out_spec = InputSpec(ndim=3, axes={-1: n_h, -2: n_in})
            b_out_spec = InputSpec(ndim=2, axes={-1: n_h})
            w_b_spec = InputSpec(ndim=3, axes={-1: n_out, -2: n_h})
            b_b_spec = InputSpec(ndim=2, axes={-1: n_out})
            self.input_spec = [w_out_spec, b_out_spec] * 2 + [w_b_spec, b_b_spec] * 2  #

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        return merge_with_previous(inputs)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "input_shape_layer": self.input_shape_layer,
                "backward_shape_layer": self.backward_shape_layer,
            }
        )
        return config


def merge_with_previous(inputs: List[keras.KerasTensor]) -> List[keras.KerasTensor]:
    w_u_out, b_u_out, w_l_out, b_l_out, w_b_u, b_b_u, w_b_l, b_b_l = inputs

    # w_u_out (None, n_h_in, n_h_out)
    # w_b_u (None, n_h_out, n_out)

    # w_u_out_ (None, n_h_in, n_h_out, 1)
    # w_b_u_ (None, 1, n_h_out, n_out)
    # w_u_out_*w_b_u_ (None, n_h_in, n_h_out, n_out)

    # result (None, n_h_in, n_out)

    if len(w_u_out.shape) == 2:
        w_u_out = BatchedDiagLike()(w_u_out)

    if len(w_l_out.shape) == 2:
        w_l_out = BatchedDiagLike()(w_l_out)

    if len(w_b_u.shape) == 2:
        w_b_u = BatchedDiagLike()(w_b_u)

    if len(w_b_l.shape) == 2:
        w_b_l = BatchedDiagLike()(w_b_l)

    # import pdb; pdb.set_trace()

    z_value = K.cast(0.0, dtype=w_u_out.dtype)
    w_b_u_pos = K.maximum(w_b_u, z_value)
    w_b_u_neg = K.minimum(w_b_u, z_value)
    w_b_l_pos = K.maximum(w_b_l, z_value)
    w_b_l_neg = K.minimum(w_b_l, z_value)

    w_u = batch_dot(w_u_out, w_b_u_pos, (-1, -2)) + batch_dot(w_l_out, w_b_u_neg, (-1, -2))
    w_l = batch_dot(w_l_out, w_b_l_pos, (-1, -2)) + batch_dot(w_u_out, w_b_l_neg, (-1, -2))
    b_u = batch_dot(b_u_out, w_b_u_pos, (-1, -2)) + batch_dot(b_l_out, w_b_u_neg, (-1, -2)) + b_b_u
    b_l = batch_dot(b_l_out, w_b_l_pos, (-1, -2)) + batch_dot(b_u_out, w_b_l_neg, (-1, -2)) + b_b_l

    return [w_u, b_u, w_l, b_l]
