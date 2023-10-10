from typing import Any, Dict, List, Optional, Tuple, Union

import keras_core as keras
import numpy as np
import tensorflow as tf
from keras_core import backend as K
from keras_core.constraints import Constraint
from keras_core.initializers import Initializer
from keras_core.layers import Layer

from decomon.core import (
    BallDomain,
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    Slope,
    get_affine,
    get_ibp,
)
from decomon.utils import add, get_linear_softplus_hull, maximum, minimum, minus, relu_


def is_a_merge_layer(layer: Layer) -> bool:
    return hasattr(layer, "_merge_function")


class NonPos(Constraint):
    """Constrains the weights to be non-negative."""

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return K.minimum(w, 0.0)


class NonNeg(Constraint):
    """Constrains the weights to be non-negative."""

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return K.maximum(w, 0.0)


class ClipAlpha(Constraint):
    """Cosntraints the weights to be between 0 and 1."""

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return K.clip(w, 0.0, 1.0)


class ClipAlphaGrid(Constraint):
    """Cosntraints the weights to be between 0 and 1."""

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        w = K.clip(w, 0.0, 1.0)
        w /= K.maximum(K.sum(w, 0), 1.0)[None]
        return w


class ClipAlphaAndSumtoOne(Constraint):
    """Cosntraints the weights to be between 0 and 1."""

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        w = K.clip(w, 0.0, 1.0)
        # normalize the first colum to 1
        w_scale = K.maximum(K.sum(w, 0), K.epsilon())
        return w / w_scale[:, None, None, None]


class MultipleConstraint(Constraint):
    """stacking multiple constraints"""

    def __init__(self, constraint_0: Optional[Constraint], constraint_1: Constraint, **kwargs: Any):
        super().__init__(**kwargs)
        if constraint_0:
            self.constraints = [constraint_0, constraint_1]
        else:
            self.constraints = [constraint_1]

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        for c in self.constraints:
            w = c.__call__(w)

        return w


class Project_initializer_pos(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, initializer: Initializer, **kwargs: Any):
        super().__init__(**kwargs)
        self.initializer = initializer

    def __call__(self, shape: tf.TensorShape, dtype: Optional[tf.DType] = None, **kwargs: Any) -> tf.Tensor:
        w = self.initializer.__call__(shape, dtype)
        return K.maximum(0.0, w)


class Project_initializer_neg(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, initializer: Initializer, **kwargs: Any):
        super().__init__(**kwargs)
        self.initializer = initializer

    def __call__(self, shape: tf.TensorShape, dtype: Optional[tf.DType] = None, **kwargs: Any) -> tf.Tensor:
        w = self.initializer.__call__(shape, dtype)
        return K.minimum(0.0, w)


def softplus_(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    u_c_out = K.softplus(u_c)
    l_c_out = K.softplus(l_c)

    if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_linear_softplus_hull(upper=u_c, lower=l_c, slope=slope, **kwargs)
        b_u_out = w_u_out * b_u + b_u_out
        b_l_out = w_l_out * b_l + b_l_out
        w_u_out = K.expand_dims(w_u_out, 1) * w_u
        w_l_out = K.expand_dims(w_l_out, 1) * w_l
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def sum(
    inputs: List[tf.Tensor],
    axis: int = -1,
    dc_decomp: bool = False,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    perturbation_domain: Optional[PerturbationDomain] = None,
    **kwargs: Any,
) -> List[tf.Tensor]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    ibp = get_ibp(mode)
    affine = get_affine(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, compute_ibp_from_affine=False
    )
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    if ibp:
        u_c_out = K.sum(u_c, axis=axis)
        l_c_out = K.sum(l_c, axis=axis)
    else:
        u_c_out, l_c_out = empty_tensor, empty_tensor

    if affine:
        if axis == -1:
            axis_w = -1
        else:
            axis_w = axis + 1
        w_u_out = K.sum(w_u, axis=axis_w)
        w_l_out = K.sum(w_l, axis=axis_w)
        b_u_out = K.sum(b_u, axis=axis)
        b_l_out = K.sum(b_l, axis=axis)
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def frac_pos(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    u_c_out = 1.0 / l_c
    l_c_out = 1.0 / u_c

    if affine:
        w_u_0 = (u_c_out - l_c_out) / K.maximum(u_c - l_c, K.epsilon())
        b_u_0 = l_c_out - w_u_0 * l_c

        y = (u_c + l_c) / 2.0
        b_l_0 = 2.0 / y
        w_l_0 = -1 / y**2

        w_u_out = w_u_0[:, None] * w_l
        b_u_out = b_u_0 * b_l + b_u_0
        w_l_out = w_l_0[:, None] * w_u
        b_l_out = b_l_0 * b_u + b_l_0
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


# convex hull of the maximum between two functions
def max_(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    finetune: bool = False,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA implementation of max(x, axis)

    Args:
        inputs: list of tensors
        dc_decomp: boolean that indicates
        perturbation_domain: the type of perturbation domain
        axis: axis to perform the maximum
    whether we return a difference of convex decomposition of our layer

    Returns:
        max operation  along an axis
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    inputs_outputs_spec_no_dc = InputsOutputsSpec(dc_decomp=False, mode=mode, perturbation_domain=perturbation_domain)

    input_shape = K.int_shape(inputs[-1])
    max_dim = input_shape[axis]
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, tight=False, compute_ibp_from_affine=False
    )
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)
    empty_tensors_list = [empty_tensor] * max_dim

    if mode == ForwardMode.IBP and not dc_decomp:
        u_c_out = K.max(u_c, axis=axis)
        l_c_out = K.max(l_c, axis=axis)

        return [u_c_out, l_c_out]

    # do some transpose so that the last axis is also at the end

    if ibp:
        u_c_list = tf.split(u_c, max_dim, axis)
        l_c_list = tf.split(l_c, max_dim, axis)
        u_c_tmp = u_c_list[0] + 0 * (u_c_list[0])
        l_c_tmp = l_c_list[0] + 0 * (l_c_list[0])
    else:
        u_c_tmp, l_c_tmp = empty_tensor, empty_tensor
        u_c_list, l_c_list = empty_tensors_list, empty_tensors_list

    if affine:
        b_u_list = tf.split(b_u, max_dim, axis)
        b_l_list = tf.split(b_l, max_dim, axis)
        b_u_tmp = b_u_list[0] + 0 * (b_u_list[0])
        b_l_tmp = b_l_list[0] + 0 * (b_l_list[0])

        if axis == -1:
            w_u_list = tf.split(w_u, max_dim, axis)
            w_l_list = tf.split(w_l, max_dim, axis)
        else:
            w_u_list = tf.split(w_u, max_dim, axis + 1)
            w_l_list = tf.split(w_l, max_dim, axis + 1)
        w_u_tmp = w_u_list[0] + 0 * (w_u_list[0])
        w_l_tmp = w_l_list[0] + 0 * (w_l_list[0])

        if finetune:
            params = kwargs["finetune_params"]
            params_split = [e[0] for e in tf.split(params[None], max_dim, axis)]
        else:
            params_split = [empty_tensor] * max_dim

    else:
        w_u_tmp, b_u_tmp, w_l_tmp, b_l_tmp = empty_tensor, empty_tensor, empty_tensor, empty_tensor
        w_u_list, b_u_list, w_l_list, b_l_list = (
            empty_tensors_list,
            empty_tensors_list,
            empty_tensors_list,
            empty_tensors_list,
        )
        params_split = [False] * max_dim

    h_tmp, g_tmp = None, None
    output_tmp = inputs_outputs_spec_no_dc.extract_outputsformode_from_fulloutputs(
        [
            x,
            u_c_tmp,
            w_u_tmp,
            b_u_tmp,
            l_c_tmp,
            w_l_tmp,
            b_l_tmp,
            h_tmp,
            g_tmp,
        ]
    )
    for i in range(1, max_dim):
        output_i = inputs_outputs_spec_no_dc.extract_outputsformode_from_fulloutputs(
            [x, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i], None, None]
        )
        output_tmp = maximum(
            output_tmp,
            output_i,
            dc_decomp=False,
            mode=mode,
            finetune=finetune,
            finetune_params=params_split[i],
            perturbation_domain=perturbation_domain,
        )

    # reduce the dimension
    tight = not (mode == ForwardMode.AFFINE)  # no need to compute u_c, l_c for pure affine mode
    (
        _,
        u_c_out,
        w_u_out,
        b_u_out,
        l_c_out,
        w_l_out,
        b_l_out,
        _,
        _,
    ) = inputs_outputs_spec_no_dc.get_fullinputs_from_inputsformode(output_tmp, tight=tight)

    if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
        u_c_out = K.squeeze(u_c_out, axis)
        l_c_out = K.squeeze(l_c_out, axis)
    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        b_u_out = K.squeeze(b_u_out, axis)
        b_l_out = K.squeeze(b_l_out, axis)
        if axis == -1:
            w_u_out = K.squeeze(w_u_out, axis)
            w_l_out = K.squeeze(w_l_out, axis)
        else:
            w_u_out = K.squeeze(w_u_out, axis + 1)
            w_l_out = K.squeeze(w_l_out, axis + 1)

    if dc_decomp:
        g_out = K.sum(g, axis=axis)
        h_out = K.max(h + g, axis=axis) - g_out
    else:
        g_out, h_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def softmax_to_linear(model: keras.Model) -> Tuple[keras.Model, bool]:
    """linearize the softmax layer for verification

    Args:
        model: Keras Model

    Returns:
        model without the softmax
    """
    layer = model.layers[-1]
    # check that layer is not an instance of the object Softmax
    if isinstance(layer, keras.layers.Softmax):
        model_normalize = keras.models.Model(model.get_input_at(0), keras.layers.Activation("linear")(layer.input))

        return model_normalize, True

    if hasattr(layer, "activation"):
        if not layer.get_config()["activation"] == "softmax":
            return model, False
        layer.get_config()["activation"] = "linear"
        layer.activation = keras.activations.get("linear")
        return model, True

    return model, False


def linear_to_softmax(model: keras.Model) -> Tuple[keras.Model, bool]:
    model.layers[-1].activation = keras.activations.get("softmax")
    return model


def multiply(
    inputs_0: List[tf.Tensor],
    inputs_1: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[tf.Tensor]:
    """LiRPA implementation of (element-wise) multiply(x,y)=-x*y.

    Args:
        inputs_0: list of tensors
        inputs_1: list of tensors
        dc_decomp: boolean that indicates
        perturbation_domain: the type of perturbation domain
        mode: type of Forward propagation (ibp, affine, or hybrid)
    whether we return a difference of convex decomposition of our layer
    whether we propagate upper and lower bounds on the values of the gradient

    Returns:
        maximum(inputs_0, inputs_1)
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, h_0, g_0 = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs_0
    )
    x_1, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1, h_1, g_1 = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs_1
    )
    dtype = x_0.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    # using McCormick's inequalities to derive bounds
    # xy<= x_u*y + x*y_L - xU*y_L
    # xy<= x*y_u + x_L*y - x_L*y_U

    # xy >=x_L*y + x*y_L -x_L*y_L
    # xy >= x_U*y + x*y_U - x_U*y_U

    if ibp:
        u_c_0_out = K.maximum(u_c_0 * u_c_1, u_c_0 * l_c_1) + K.maximum(u_c_0 * l_c_1, l_c_0 * l_c_1) - u_c_0 * l_c_1
        u_c_1_out = K.maximum(u_c_1 * u_c_0, u_c_1 * l_c_0) + K.maximum(u_c_1 * l_c_0, l_c_1 * l_c_0) - u_c_1 * l_c_0
        u_c_out = K.minimum(u_c_0_out, u_c_1_out)
        l_c_out = K.minimum(l_c_0 * l_c_1, l_c_0 * u_c_1) + K.minimum(l_c_0 * l_c_1, u_c_0 * l_c_1) - l_c_0 * l_c_1
    else:
        u_c_out, l_c_out = empty_tensor, empty_tensor

    if affine:
        # xy <= x_u * y + x * y_L - xU * y_L
        cx_u_pos = K.maximum(u_c_0, 0.0)
        cx_u_neg = K.minimum(u_c_0, 0.0)

        cy_l_pos = K.maximum(l_c_1, 0.0)
        cy_l_neg = K.minimum(l_c_1, 0.0)
        w_u_out = (
            cx_u_pos[:, None] * w_u_1
            + cx_u_neg[:, None] * w_l_1
            + cy_l_pos[:, None] * w_u_0
            + cy_l_neg[:, None] * w_l_0
        )
        b_u_out = cx_u_pos * b_u_1 + cx_u_neg * b_l_1 + cy_l_pos * b_u_0 + cy_l_neg * b_l_0 - u_c_0 * l_c_1

        # xy >= x_U*y + x*y_U - x_U*y_U
        cx_l_pos = K.maximum(l_c_0, 0.0)
        cx_l_neg = K.minimum(l_c_0, 0.0)

        w_l_out = (
            cx_l_pos[:, None] * w_l_1
            + cx_l_neg[:, None] * w_u_1
            + cy_l_pos[:, None] * w_l_0
            + cy_l_neg[:, None] * w_u_0
        )
        b_l_out = cx_l_pos * b_l_1 + cx_l_neg * b_u_1 + cy_l_pos * b_l_0 + cy_l_neg * b_u_0 - l_c_0 * l_c_1
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def permute_dimensions(
    inputs: List[tf.Tensor],
    axis: int,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis_perm: int = 1,
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
) -> List[tf.Tensor]:
    """LiRPA implementation of (element-wise) permute(x,axis)

    Args:
        inputs: list of input tensors
        axis: axis on which we apply the permutation
        mode: type of Forward propagation (ibp, affine, or hybrid)
        axis_perm: see DecomonPermute operator

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, tight=False, compute_ibp_from_affine=False
    )
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    input_shape = inputs_outputs_spec.get_input_shape(inputs)
    if len(input_shape) <= 2:
        # not enough dim to permute
        return inputs
    index = np.arange(len(input_shape))
    index = np.insert(np.delete(index, axis), axis_perm, axis)
    index_w = np.arange(len(input_shape) + 1)
    index_w = np.insert(np.delete(index_w, axis), axis_perm + 1, axis)

    if ibp:
        u_c_out = K.permute_dimensions(u_c, index)
        l_c_out = K.permute_dimensions(l_c, index)
    else:
        u_c_out, l_c_out = empty_tensor, empty_tensor
    if affine:
        w_u_out = K.permute_dimensions(w_u, index_w)
        b_u_out = K.permute_dimensions(b_u, index)
        w_l_out = K.permute_dimensions(w_l, index_w)
        b_l_out = K.permute_dimensions(b_l, index)
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def broadcast(
    inputs: List[tf.Tensor],
    n: int,
    axis: int,
    mode: Union[str, ForwardMode],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
) -> List[tf.Tensor]:
    """LiRPA implementation of broadcasting

    Args:
        inputs
        n
        axis
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, tight=False, compute_ibp_from_affine=False
    )
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    if ibp:
        for _ in range(n):
            u_c = K.expand_dims(u_c, axis)
            l_c = K.expand_dims(l_c, axis)

    if axis != -1:
        axis_w = axis + 1
    else:
        axis_w = -1

    if affine:
        for _ in range(n):
            b_u = K.expand_dims(b_u, axis)
            b_l = K.expand_dims(b_l, axis)
            w_u = K.expand_dims(w_u, axis_w)
            w_l = K.expand_dims(w_l, axis_w)

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs([x, u_c, w_u, b_u, l_c, w_l, b_l, h, g])


def split(
    inputs: List[tf.Tensor],
    axis: int = -1,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
) -> List[List[tf.Tensor]]:
    """LiRPA implementation of split

    Args:
        inputs
        axis
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    input_shape = inputs_outputs_spec.get_input_shape(inputs)
    if axis == -1:
        n = input_shape[-1]
        axis = len(input_shape) - 1
    else:
        n = input_shape[axis]
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, tight=False, compute_ibp_from_affine=False
    )
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)
    empty_tensor_list = [empty_tensor] * n

    if ibp:
        u_c_list = tf.split(u_c, num_or_size_splits=n, axis=axis)
        l_c_list = tf.split(l_c, num_or_size_splits=n, axis=axis)
    else:
        u_c_list, l_c_list = empty_tensor_list, empty_tensor_list

    if affine:
        b_u_list = tf.split(b_u, num_or_size_splits=n, axis=axis)
        b_l_list = tf.split(b_l, num_or_size_splits=n, axis=axis)

        if axis != -1:
            axis += 1
        w_u_list = tf.split(w_u, num_or_size_splits=n, axis=axis)
        w_l_list = tf.split(w_l, num_or_size_splits=n, axis=axis)
    else:
        w_u_list, b_u_list, w_l_list, b_l_list = (
            empty_tensor_list,
            empty_tensor_list,
            empty_tensor_list,
            empty_tensor_list,
        )

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_list, g_list = empty_tensor_list, empty_tensor_list

    return [
        inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i], h_list[i], g_list[i]]
        )
        for i in range(n)
    ]


def sort(
    inputs: List[tf.Tensor],
    axis: int = -1,
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[tf.Tensor]:
    """LiRPA implementation of sort by selection

    Args:
        inputs
        axis
        dc_decomp
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, tight=False, compute_ibp_from_affine=False
    )
    input_shape = inputs_outputs_spec.get_input_shape(inputs)
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    if axis == -1:
        n = input_shape[-1]
        axis = len(input_shape) - 1
    else:
        n = input_shape[axis]

    empty_tensor_list = [empty_tensor] * n

    # what about splitting elements
    op_split = lambda x: tf.split(x, n, axis=axis)
    if ibp:
        u_c_list = op_split(u_c)
        l_c_list = op_split(l_c)
    else:
        u_c_list, l_c_list = empty_tensor_list, empty_tensor_list

    if affine:
        w_u_list = tf.split(w_u, n, axis=axis + 1)
        b_u_list = op_split(b_u)
        w_l_list = tf.split(w_l, n, axis=axis + 1)
        b_l_list = op_split(b_l)
    else:
        w_u_list, b_u_list, w_l_list, b_l_list = (
            empty_tensor_list,
            empty_tensor_list,
            empty_tensor_list,
            empty_tensor_list,
        )

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_list, g_list = empty_tensor_list, empty_tensor_list

    # use selection sort
    for i in range(n - 1):
        for j in range(i + 1, n):
            input_i = inputs_outputs_spec.extract_inputsformode_from_fullinputs(
                [x, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i], h_list[i], g_list[i]]
            )
            input_j = inputs_outputs_spec.extract_inputsformode_from_fullinputs(
                [x, u_c_list[j], w_u_list[j], b_u_list[j], l_c_list[j], w_l_list[j], b_l_list[j], h_list[j], g_list[j]]
            )

            # max and min of splitted inputs
            output_a = maximum(
                input_i, input_j, mode=mode, perturbation_domain=perturbation_domain, dc_decomp=dc_decomp
            )
            output_b = minimum(
                input_i, input_j, mode=mode, perturbation_domain=perturbation_domain, dc_decomp=dc_decomp
            )

            # update lists
            (
                x,
                u_c_list[j],
                w_u_list[j],
                b_u_list[j],
                l_c_list[j],
                w_l_list[j],
                b_l_list[j],
                h_list[j],
                g_list[j],
            ) = inputs_outputs_spec.get_fullinputs_from_inputsformode(output_a)
            (
                x,
                u_c_list[i],
                w_u_list[i],
                b_u_list[i],
                l_c_list[i],
                w_l_list[i],
                b_l_list[i],
                h_list[i],
                g_list[i],
            ) = inputs_outputs_spec.get_fullinputs_from_inputsformode(output_b)

    op_concatenate = lambda x: K.concatenate(x, axis)
    # update the inputs
    if ibp:
        u_c_out = op_concatenate(u_c_list)
        l_c_out = op_concatenate(l_c_list)
    else:
        u_c_out, l_c_out = empty_tensor, empty_tensor
    if affine:
        w_u_out = K.concatenate(w_u_list, axis + 1)
        w_l_out = K.concatenate(w_l_list, axis + 1)
        b_u_out = op_concatenate(b_u_list)
        b_l_out = op_concatenate(b_l_list)
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def pow(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[tf.Tensor]:
    """LiRPA implementation of pow(x )=x**2

    Args:
        inputs
        dc_decomp
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    return multiply(inputs, inputs, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode)


def abs(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[tf.Tensor]:
    """LiRPA implementation of |x|

    Args:
        inputs
        dc_decomp
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    inputs_0 = relu_(inputs, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode)
    inputs_1 = minus(
        relu_(
            minus(inputs, mode=mode, perturbation_domain=perturbation_domain),
            dc_decomp=dc_decomp,
            perturbation_domain=perturbation_domain,
            mode=mode,
        ),
        mode=mode,
        perturbation_domain=perturbation_domain,
    )

    return add(inputs_0, inputs_1, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode)


def frac_pos_hull(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[tf.Tensor]:
    """LiRPA implementation of 1/x for x>0

    Args:
        inputs
        dc_decomp
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)

    l_c = K.maximum(l_c, 1.0)
    z = (u_c + l_c) / 2.0
    w_l = -1 / K.pow(z)
    b_l = 2 / z
    w_u = (1.0 / u_c - 1.0 / l_c) / (u_c - l_c)
    b_u = 1.0 / u_c - w_u * u_c

    return [w_u, b_u, w_l, b_l]


# convex hull for min
def min_(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    finetune: bool = False,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA implementation of min(x, axis=axis)

    Args:
        inputs
        dc_decomp
        perturbation_domain
        mode
        axis

    Returns:

    """
    # return - max - x
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    return minus(
        max_(
            minus(inputs, mode=mode, perturbation_domain=perturbation_domain),
            dc_decomp=dc_decomp,
            perturbation_domain=perturbation_domain,
            mode=mode,
            axis=axis,
            finetune=finetune,
            **kwargs,
        ),
        mode=mode,
        perturbation_domain=perturbation_domain,
    )


def expand_dims(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    perturbation_domain: Optional[PerturbationDomain] = None,
    **kwargs: Any,
) -> List[tf.Tensor]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, tight=False, compute_ibp_from_affine=False
    )
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    if axis == -1:
        axis_w = axis
    else:
        axis_w = axis + 1

    op = lambda t: K.expand_dims(t, axis)

    if ibp:
        u_c_out = op(u_c)
        l_c_out = op(l_c)
    else:
        u_c_out, l_c_out = empty_tensor, empty_tensor

    if affine:
        b_u_out = op(b_u)
        b_l_out = op(b_l)
        w_u_out = K.expand_dims(w_u, axis_w)
        w_l_out = K.expand_dims(w_l, axis_w)
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def log(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Exponential activation function.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: the type of convex input domain
        mode: type of Forward propagation (ibp, affine, or hybrid)
        **kwargs: extra parameters
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    if mode == ForwardMode.AFFINE:
        l_c = K.maximum(K.cast(K.epsilon(), dtype=l_c.dtype), l_c)

    u_c_out = K.log(u_c)
    l_c_out = K.log(l_c)

    if affine:
        y = (u_c + l_c) / 2.0

        w_l_0 = (u_c_out - l_c_out) / K.maximum(u_c - l_c, K.epsilon())
        b_l_0 = l_c_out - w_l_0 * l_c

        w_u_0 = 1 / y
        b_u_0 = K.log(y) - 1

        w_u_out = w_u_0[:, None] * w_u
        b_u_out = w_u_0 * b_u + b_u_0
        w_l_out = w_l_0[:, None] * w_l
        b_l_out = w_l_0 * b_l + b_l_0
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def exp(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Exponential activation function.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: the type of convex input domain
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: extra parameters
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    u_c_out = K.exp(u_c)
    l_c_out = K.exp(l_c)

    if affine:
        y = (u_c + l_c) / 2.0  # do finetuneting

        w_u_0 = (u_c_out - l_c_out) / K.maximum(u_c - l_c, K.epsilon())
        b_u_0 = l_c_out - w_u_0 * l_c

        w_l_0 = K.exp(y)
        b_l_0 = w_l_0 * (1 - y)

        w_u_out = w_u_0[:, None] * w_u
        b_u_out = w_u_0 * b_u + b_u_0
        w_l_out = w_l_0[:, None] * w_l
        b_l_out = w_l_0 * b_l + b_l_0
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError("Not yet implemented for dc_decomp=True")
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )
