from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Layer

from decomon.core import (
    BallDomain,
    BoxDomain,
    InputsOutputsSpec,
    PerturbationDomain,
    Slope,
)
from decomon.layers.core import ForwardMode
from decomon.utils import (
    add,
    get_linear_softplus_hull,
    get_lower,
    get_upper,
    maximum,
    minimum,
    minus,
    relu_,
)


def is_a_merge_layer(layer: Layer) -> bool:
    return hasattr(layer, "_merge_function")


#####
# USE SYMBOLIC GRADIENT DESCENT WITH OVERESTIMATION GUARANTEES
#####
# first compute gradient of the function
def get_grad(x: tf.Tensor, constant: tf.Tensor, W: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """We compute the gradient of the function f at sample x

    f = sum_{i<= n_linear} max(constant_i, W_i*x + b_i)
    it is quite easy to compute the gradient symbolically, without using the gradient operator
    it is either 0 if f(x) = constant or W else
    this trick allows to be backpropagation compatible

    Args:
        x: Keras Tensor (None, n_dim, ...)
        constant: Keras Tensor (None, n_linear, ...)
        W: Keras Tensor (None, n_dim, n_linear, ...)
        b: Keras Tensor (None, n_linear, ...)

    Returns:
        Keras Tensor the gradient of the function (None, n_dim, ...)
    """
    # product W*x + b
    x_reshaped = K.expand_dims(x, 2)
    z = K.sum(W * x_reshaped, 1) + b
    return K.sum(K.expand_dims(-K.sign(constant - K.maximum(constant, z)), 1) * W, 2)  # (None, n_dim, ...)


def compute_L(W: tf.Tensor) -> tf.Tensor:
    """We compute the largest possible norm of the gradient

    Args:
        W: Keras Tensor (None, n_dim, n_linear, ...)

    Returns:
        Keras Tensor with an upper bound on the largest magnitude of the
        gradient
    """
    return K.sum(K.sqrt(K.sum(W * W, 1)), 1)


def compute_R(z: tf.Tensor, perturbation_domain: PerturbationDomain) -> tf.Tensor:
    """We compute the largest L2 distance of the starting point with the global optimum

    Args:
        z: Keras Tensor
        perturbation_domain: Dictionnary to complement z on the perturbation domain

    Returns:
        Keras Tensor an upper bound on the distance
    """
    if isinstance(perturbation_domain, BoxDomain):
        return K.sqrt(K.sum(K.pow((z[:, 1] - z[:, 0]) / 2.0, 2), -1))
    elif isinstance(perturbation_domain, BallDomain):
        if perturbation_domain.p == np.inf:
            return K.sqrt(K.sum(K.pow(z - z + perturbation_domain.eps, 2), -1))
        elif perturbation_domain.p == 2:
            return perturbation_domain.eps * (0 * z + 1.0)
        else:
            raise NotImplementedError(f"Not implemented for ball domain type with p={perturbation_domain.p}")
    else:
        raise NotImplementedError(f"Not implemented for perturbation domain type {type(perturbation_domain)}")


def get_start_point(z: tf.Tensor, perturbation_domain: PerturbationDomain) -> tf.Tensor:
    """Create a warm start for the optimization (the mid point to minimize the largest distance between the
    warm start and the global optimum

    Args:
        z: Keras Tensor
        perturbation_domain: Dictionnary to complement z on the perturbation domain

    Returns:
        Keras Tensor
    """

    if isinstance(perturbation_domain, BoxDomain):
        return (z[:, 0] + z[:, 1]) / 2.0
    elif isinstance(perturbation_domain, BallDomain):
        return z
    else:
        raise NotImplementedError(f"Not implemented for perturbation domain type {type(perturbation_domain)}")


def get_coeff_grad(R: tf.Tensor, k: int, g: tf.Tensor) -> tf.Tensor:
    """
    Args:
        R: Keras Tensor that reprends the largest distance to the gloabl
            optimum
        k: the number of iteration done so far
        g: the gradient

    Returns:
        the adaptative step size for the gradient
    """

    denum = np.sqrt(k) * K.sqrt(K.sum(K.pow(g, 2), 1))

    alpha = R / K.maximum(K.epsilon(), denum)
    return alpha


def grad_descent_conv(
    z: tf.Tensor,
    concave_upper: List[tf.Tensor],
    convex_lower: List[tf.Tensor],
    op_pos: List[tf.Tensor],
    ops_neg: List[tf.Tensor],
    n_iter: int = 5,
) -> tf.Tensor:
    """
    Args:
        z
        concave_upper
        convex_lower
        op_pos
        ops_neg
        n_iter

    Returns:

    """

    raise NotImplementedError()


def grad_descent(
    z: tf.Tensor,
    convex_0: List[tf.Tensor],
    convex_1: List[tf.Tensor],
    perturbation_domain: PerturbationDomain,
    n_iter: int = 5,
) -> tf.Tensor:
    """
    Args:
        z: Keras Tensor
        constant: Keras Tensor, the constant of each component
        W: Keras Tensor, the affine of each component
        b: Keras Tensor, the bias of each component
        perturbation_domain: Dictionnary to complement z on the perturbation domain
        n_iter: the number of total iteration

    Returns:

    """

    constant_0, W_0, b_0 = convex_0
    constant_1, W_1, b_1 = convex_1

    # init
    x_k = K.expand_dims(get_start_point(z, perturbation_domain), -1) + 0 * K.sum(constant_0, 1)[:, None]
    R = compute_R(z, perturbation_domain)
    n_dim = len(x_k.shape[1:])
    while n_dim > 1:
        R = K.expand_dims(R, -1)
        n_dim -= 1

    def step_grad(x: tf.Tensor, inputs_k: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:

        x_k = inputs_k[0]
        g_k_0 = get_grad(x_k, constant_0, W_0, b_0)
        g_k_1 = get_grad(x_k, constant_1, W_1, b_1)
        g_k = g_k_0 + g_k_1
        alpha_k = get_coeff_grad(R, n_iter + 1, g_k)
        x_result = alpha_k[:, None] * g_k
        return x_result, [x_k]

    x_vec = K.rnn(
        step_function=step_grad, inputs=K.concatenate([x_k[:, None]] * n_iter, 1), initial_states=[x_k], unroll=False
    )[1]

    # check convergence
    x_k = x_vec[:, -1]
    g_k = get_grad(x_k, constant_0, W_0, b_0) + get_grad(x_k, constant_1, W_1, b_1)
    mask_grad = K.sign(K.sqrt(K.sum(K.pow(g_k, 2), 1)))  # check whether we have converge
    X_vec = K.expand_dims(x_vec, -2)
    f_vec = K.sum(
        K.maximum(constant_0[:, None], K.sum(W_0[:, None] * X_vec, 2) + b_0[:, None])
        + K.maximum(constant_1[:, None], K.sum(W_1[:, None] * X_vec, 2) + b_1[:, None]),
        2,
    )
    f_vec = f_vec[0]
    L_0 = compute_L(W_0)
    L_1 = compute_L(W_1)
    L = L_0 + L_1
    penalty = (L * R) / np.sqrt(n_iter + 1)
    return f_vec - mask_grad * penalty


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

    nb_tensors = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    if mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
    elif mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensors]
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if dc_decomp:
        h, g = inputs[-2:]

    if mode == ForwardMode.AFFINE:
        upper = get_upper(x_0, w_u, b_u, perturbation_domain)
        lower = get_lower(x_0, w_l, b_l, perturbation_domain)
    elif mode == ForwardMode.HYBRID:
        upper = u_c
        lower = l_c
    elif mode == ForwardMode.IBP:
        upper = u_c
        lower = l_c
    else:
        raise ValueError(f"Unknown mode {mode}")

    u_c_out = K.softplus(upper)
    l_c_out = K.softplus(lower)

    if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:

        w_u_out, b_u_out, w_l_out, b_l_out = get_linear_softplus_hull(upper=upper, lower=lower, slope=slope, **kwargs)
        b_u_out = w_u_out * b_u + b_u_out
        b_l_out = w_l_out * b_l + b_l_out
        w_u_out = K.expand_dims(w_u_out, 1) * w_u
        w_l_out = K.expand_dims(w_l_out, 1) * w_l

    if mode == ForwardMode.IBP:
        return [u_c_out, l_c_out]
    elif mode == ForwardMode.AFFINE:
        return [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
    elif mode == ForwardMode.HYBRID:
        return [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")


def sum(
    inputs: List[tf.Tensor],
    axis: int = -1,
    dc_decomp: bool = False,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:

    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    if mode == ForwardMode.IBP:
        return [K.sum(inputs[0], axis=axis), K.sum(inputs[1], axis=axis)]
    elif mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
        if mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs
        else:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs
            u_c_out = K.sum(u_c, axis=axis)
            l_c_out = K.sum(l_c, axis=axis)

        axis_w = -1
        if axis != -1:
            axis_w = axis + 1
        w_u_out = K.sum(w_u, axis=axis_w)
        w_l_out = K.sum(w_l, axis=axis_w)
        b_u_out = K.sum(b_u, axis=axis)
        b_l_out = K.sum(b_l, axis=axis)

        if mode == ForwardMode.AFFINE:
            return [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
        else:
            return [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")


def frac_pos(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    # frac_pos is convex for positive values
    if mode == ForwardMode.IBP:
        u_c, l_c = inputs
        u_c_out = 1.0 / l_c
        l_c_out = 1.0 / u_c
        return [u_c_out, l_c_out]
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs
        u_c = get_upper(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
        l_c = get_lower(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
    elif mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs
    else:
        raise ValueError(f"Mode {mode} unknown")

    u_c_out = 1.0 / l_c
    l_c_out = 1.0 / u_c

    w_u_0 = (u_c_out - l_c_out) / K.maximum(u_c - l_c, K.epsilon())
    b_u_0 = l_c_out - w_u_0 * l_c

    y = (u_c + l_c) / 2.0
    b_l_0 = 2.0 / y
    w_l_0 = -1 / y**2

    w_u_out = w_u_0[:, None] * w_l
    b_u_out = b_u_0 * b_l + b_u_0
    w_l_out = w_l_0[:, None] * w_u
    b_l_out = b_l_0 * b_u + b_l_0

    if mode == ForwardMode.AFFINE:
        return [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
    elif mode == ForwardMode.HYBRID:
        return [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Mode {mode} unknown")


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
    if dc_decomp:
        h, g = inputs[-2:]
    mode = ForwardMode(mode)
    nb_tensor = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors

    if mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensor]
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    elif mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode == ForwardMode.IBP and not dc_decomp:
        u_c_out = K.max(u_c, axis=axis)
        l_c_out = K.max(l_c, axis=axis)

        return [u_c_out, l_c_out]

    input_shape = K.int_shape(inputs[-1])
    max_dim = input_shape[axis]

    # do some transpose so that the last axis is also at the end

    if dc_decomp:
        h_list = tf.split(h, max_dim, axis)
        g_list = tf.split(g, max_dim, axis)

    if mode in [ForwardMode.HYBRID, ForwardMode.IBP]:

        u_c_list = tf.split(u_c, max_dim, axis)
        l_c_list = tf.split(l_c, max_dim, axis)
        u_c_tmp = u_c_list[0] + 0 * (u_c_list[0])
        l_c_tmp = l_c_list[0] + 0 * (l_c_list[0])

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:

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
            key = [e for e in kwargs.keys()][0]
            params = kwargs[key][0]
            params_split = [e[0] for e in tf.split(params[None], max_dim, axis)]

    output_tmp = []
    if mode == ForwardMode.HYBRID:
        output_tmp = [
            x_0,
            u_c_tmp,
            w_u_tmp,
            b_u_tmp,
            l_c_tmp,
            w_l_tmp,
            b_l_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]
            if finetune:
                output_tmp = maximum(
                    output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune, finetune_params=params_split[i]
                )
            else:
                output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    elif mode == ForwardMode.IBP:
        output_tmp = [
            u_c_tmp,
            l_c_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [u_c_list[i], l_c_list[i]]
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune)

    elif mode == ForwardMode.AFFINE:
        output_tmp = [
            x_0,
            w_u_tmp,
            b_u_tmp,
            w_l_tmp,
            b_l_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
            if finetune:
                output_tmp = maximum(
                    output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune, finetune_params=params_split[i]
                )
            else:
                output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # reduce the dimension
    if mode == ForwardMode.IBP:
        u_c_out, l_c_out = output_tmp[:nb_tensor]
    elif mode == ForwardMode.AFFINE:
        _, w_u_out, b_u_out, w_l_out, b_l_out = output_tmp[:nb_tensor]
    elif mode == ForwardMode.HYBRID:
        _, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out = output_tmp[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

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

    if mode == ForwardMode.HYBRID:

        upper = get_upper(x_0, w_u_out, b_u_out, perturbation_domain)
        u_c_out = K.minimum(upper, u_c_out)

        lower = get_lower(x_0, w_l_out, b_l_out, perturbation_domain)
        l_c_out = K.maximum(lower, l_c_out)

    if mode == ForwardMode.HYBRID:
        output = [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    elif mode == ForwardMode.IBP:
        output = [u_c_out, l_c_out]
    elif mode == ForwardMode.AFFINE:
        output = [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if dc_decomp:
        output += [h_out, g_out]

    return output


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
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    nb_tensor = InputsOutputsSpec(dc_decomp, mode=mode).nb_tensors

    if mode == ForwardMode.IBP:
        u0, l0 = inputs_0[:nb_tensor]
        u1, l1 = inputs_1[:nb_tensor]
    elif mode == ForwardMode.AFFINE:
        x, w_u0, b_u0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, w_u1, b_u1, w_l1, b_l1 = inputs_1[:nb_tensor]
        u0 = get_upper(x, w_u0, b_u0, perturbation_domain=perturbation_domain)
        l0 = get_lower(x, w_l0, b_l0, perturbation_domain=perturbation_domain)
        u1 = get_upper(x, w_u1, b_u1, perturbation_domain=perturbation_domain)
        l1 = get_lower(x, w_l1, b_l1, perturbation_domain=perturbation_domain)
    elif mode == ForwardMode.HYBRID:
        x, u0, w_u0, b_u0, l0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, u1, w_u1, b_u1, l1, w_l1, b_l1 = inputs_1[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")
    # using McCormick's inequalities to derive bounds
    # xy<= x_u*y + x*y_L - xU*y_L
    # xy<= x*y_u + x_L*y - x_L*y_U

    # xy >=x_L*y + x*y_L -x_L*y_L
    # xy >= x_U*y + x*y_U - x_U*y_U

    if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
        u_c_0_out = K.maximum(u0 * u1, u0 * l1) + K.maximum(u0 * l1, l0 * l1) - u0 * l1
        u_c_1_out = K.maximum(u1 * u0, u1 * l0) + K.maximum(u1 * l0, l1 * l0) - u1 * l0
        u_c_out = K.minimum(u_c_0_out, u_c_1_out)
        l_c_out = K.minimum(l0 * l1, l0 * u1) + K.minimum(l0 * l1, u0 * l1) - l0 * l1

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        # xy <= x_u * y + x * y_L - xU * y_L
        cx_u_pos = K.maximum(u0, 0.0)
        cx_u_neg = K.minimum(u0, 0.0)

        cy_l_pos = K.maximum(l1, 0.0)
        cy_l_neg = K.minimum(l1, 0.0)
        w_u_out = (
            cx_u_pos[:, None] * w_u1 + cx_u_neg[:, None] * w_l1 + cy_l_pos[:, None] * w_u0 + cy_l_neg[:, None] * w_l0
        )
        b_u_out = cx_u_pos * b_u1 + cx_u_neg * b_l1 + cy_l_pos * b_u0 + cy_l_neg * b_l0 - u0 * l1

        # xy >= x_U*y + x*y_U - x_U*y_U
        cx_l_pos = K.maximum(l0, 0.0)
        cx_l_neg = K.minimum(l0, 0.0)

        w_l_out = (
            cx_l_pos[:, None] * w_l1 + cx_l_neg[:, None] * w_u1 + cy_l_pos[:, None] * w_l0 + cy_l_neg[:, None] * w_u0
        )
        b_l_out = cx_l_pos * b_l1 + cx_l_neg * b_u1 + cy_l_pos * b_l0 + cy_l_neg * b_u0 - l0 * l1

    if mode == ForwardMode.IBP:
        return [u_c_out, l_c_out]
    elif mode == ForwardMode.AFFINE:
        return [x, w_u_out, b_u_out, w_l_out, b_l_out]
    elif mode == ForwardMode.HYBRID:
        return [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")


def permute_dimensions(
    inputs: List[tf.Tensor], axis: int, mode: Union[str, ForwardMode] = ForwardMode.HYBRID, axis_perm: int = 1
) -> List[tf.Tensor]:
    """LiRPA implementation of (element-wise) permute(x,axis)

    Args:
        inputs: list of input tensors
        axis: axis on which we apply the permutation
        mode: type of Forward propagation (ibp, affine, or hybrid)
        axis_perm: see DecomonPermute operator

    Returns:

    """

    if len(inputs[0].shape) <= 2:
        return inputs
    mode = ForwardMode(mode)
    index = np.arange(len(inputs[0].shape))
    index = np.insert(np.delete(index, axis), axis_perm, axis)
    nb_tensor = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    if mode == ForwardMode.IBP:
        return [
            K.permute_dimensions(inputs[2], index),
            K.permute_dimensions(inputs[3], index),
        ]
    else:
        index_w = np.arange(len(inputs[0].shape) + 1)
        index_w = np.insert(np.delete(index_w, axis), axis_perm + 1, axis)

        if mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
            return [
                x_0,
                K.permute_dimensions(w_u, index_w),
                K.permute_dimensions(b_u, index),
                K.permute_dimensions(w_l, index_w),
                K.permute_dimensions(b_l, index),
            ]

        elif mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs

            return [
                x_0,
                K.permute_dimensions(u_c, index),
                K.permute_dimensions(w_u, index_w),
                K.permute_dimensions(b_u, index),
                K.permute_dimensions(l_c, index),
                K.permute_dimensions(w_l, index_w),
                K.permute_dimensions(b_l, index),
            ]

        else:
            raise ValueError(f"Unknown mode {mode}")


def broadcast(inputs: List[tf.Tensor], n: int, axis: int, mode: Union[str, ForwardMode]) -> List[tf.Tensor]:
    """LiRPA implementation of broadcasting

    Args:
        inputs
        n
        axis
        mode

    Returns:

    """
    nb_tensor = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    mode = ForwardMode(mode)
    if mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensor]
    elif mode == ForwardMode.AFFINE:
        x, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    elif mode == ForwardMode.HYBRID:
        x, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
        for _ in range(n):
            u_c = K.expand_dims(u_c, axis)
            l_c = K.expand_dims(l_c, axis)

    if axis != -1:
        axis_w = axis + 1
    else:
        axis_w = -1

    if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
        for _ in range(n):
            b_u = K.expand_dims(b_u, axis)
            b_l = K.expand_dims(b_l, axis)
            w_u = K.expand_dims(w_u, axis_w)
            w_l = K.expand_dims(w_l, axis_w)

    if mode == ForwardMode.IBP:
        output = [u_c, l_c]
    elif mode == ForwardMode.AFFINE:
        output = [x, w_u, b_u, w_l, b_l]
    elif mode == ForwardMode.HYBRID:
        output = [x, u_c, w_u, b_u, l_c, w_l, b_l]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return output


def split(
    inputs: List[tf.Tensor], axis: int = -1, mode: Union[str, ForwardMode] = ForwardMode.HYBRID
) -> List[tf.Tensor]:
    """LiRPA implementation of split

    Args:
        inputs
        axis
        mode

    Returns:

    """
    mode = ForwardMode(mode)
    nb_tensor = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    if mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensor]
    elif mode == ForwardMode.HYBRID:
        x, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensor]
    elif mode == ForwardMode.AFFINE:
        x, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
        u_list = tf.split(u_c, 1, axis=axis)
        l_list = tf.split(l_c, 1, axis=axis)
        n = len(u_list)

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        b_u_list = tf.split(b_u, 1, axis=axis)
        b_l_list = tf.split(b_l, 1, axis=axis)
        n = len(b_u_list)

        if axis != -1:
            axis += 1
        w_u_list = tf.split(w_u, 1, axis=axis)
        w_l_list = tf.split(w_l, 1, axis=axis)

    if mode == ForwardMode.IBP:
        outputs = [[u_list[i], l_list[i]] for i in range(n)]
    elif mode == ForwardMode.AFFINE:
        outputs = [[x, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]
    elif mode == ForwardMode.HYBRID:
        outputs = [[x, u_list[i], w_u_list[i], b_u_list[i], l_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return outputs


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
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    # remove grad bounds

    nb_tensor = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors

    if mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensor]
    elif mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensor]
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if axis == -1:
        n = inputs[-1].shape[-1]
        axis = len(inputs[-1].shape) - 1
    else:
        n = inputs[-1].shape[axis]

    # what about splitting elements
    op_split = lambda x: tf.split(x, n, axis=axis)
    if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
        u_c_list = op_split(u_c)
        l_c_list = op_split(l_c)

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        w_u_list = tf.split(w_u, n, axis=axis + 1)
        b_u_list = op_split(b_u)
        w_l_list = tf.split(w_l, n, axis=axis + 1)
        b_l_list = op_split(b_l)

    def get_input(mode: ForwardMode, i: int) -> List[tf.Tensor]:
        if mode == ForwardMode.IBP:
            return [u_c_list[i], l_c_list[i]]
        elif mode == ForwardMode.AFFINE:
            return [x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
        elif mode == ForwardMode.HYBRID:
            return [x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]
        else:
            raise ValueError(f"Unknown mode {mode}")

    def set_input(inputs: List[tf.Tensor], mode: ForwardMode, i: int) -> None:
        if mode == ForwardMode.IBP:
            u_i, l_i = inputs
        elif mode == ForwardMode.AFFINE:
            _, w_u_i, b_u_i, w_l_i, b_l_i = inputs
        elif mode == ForwardMode.HYBRID:
            _, u_i, w_u_i, b_u_i, l_i, w_l_i, b_l_i = inputs
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_list[i] = u_i
            l_c_list[i] = l_i
        if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            w_u_list[i] = w_u_i
            w_l_list[i] = w_l_i
            b_u_list[i] = b_u_i
            b_l_list[i] = b_l_i

    # use selection sort
    for i in range(n - 1):
        for j in range(i + 1, n):

            input_i = get_input(mode, i)
            input_j = get_input(mode, j)
            output_a = maximum(
                input_i, input_j, mode=mode, perturbation_domain=perturbation_domain, dc_decomp=dc_decomp
            )
            output_b = minimum(
                input_i, input_j, mode=mode, perturbation_domain=perturbation_domain, dc_decomp=dc_decomp
            )

            set_input(output_a, mode, j)
            set_input(output_b, mode, i)

    op_concatenate = lambda x: K.concatenate(x, axis)
    # update the inputs
    if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
        u_c_out = op_concatenate(u_c_list)
        l_c_out = op_concatenate(l_c_list)
    if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
        w_u_out = K.concatenate(w_u_list, axis + 1)
        w_l_out = K.concatenate(w_l_list, axis + 1)
        b_u_out = op_concatenate(b_u_list)
        b_l_out = op_concatenate(b_l_list)

    if mode == ForwardMode.IBP:
        output = [u_c_out, l_c_out]
    elif mode == ForwardMode.AFFINE:
        output = [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
    elif mode == ForwardMode.HYBRID:
        output = [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return output


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
        relu_(minus(inputs, mode=mode), dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode),
        mode=mode,
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
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    nb_tensor = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors

    if mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensor]
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    elif mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")
    if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
        u_c_affine = get_upper(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
        l_c_affine = get_lower(x_0, w_u, b_u, perturbation_domain=perturbation_domain)

        if mode == ForwardMode.AFFINE:
            u_c = u_c_affine
            l_c = l_c_affine
        else:
            u_c = K.minimum(u_c_affine, u_c)
            l_c = K.maximum(l_c, l_c_affine)

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
            minus(inputs, mode=mode),
            dc_decomp=dc_decomp,
            perturbation_domain=perturbation_domain,
            mode=mode,
            axis=axis,
            finetune=finetune,
            **kwargs,
        ),
        mode=mode,
    )


def expand_dims(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    **kwargs: Any,
) -> List[tf.Tensor]:
    mode = ForwardMode(mode)
    nb_tensor = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    if mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensor]
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    elif mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        if axis == -1:
            axis_w = axis
        else:
            axis_w = axis + 1

    op = lambda t: K.expand_dims(t, axis)

    if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
        u_c_out = op(u_c)
        l_c_out = op(l_c)

    if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:

        b_u_out = op(b_u)
        b_l_out = op(b_l)
        w_u_out = K.expand_dims(w_u, axis_w)
        w_l_out = K.expand_dims(w_l, axis_w)

    if mode == ForwardMode.IBP:
        output = [u_c_out, l_c_out]
    elif mode == ForwardMode.AFFINE:
        output = [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
    elif mode == ForwardMode.HYBRID:
        output = [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")

    return output


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
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    if mode == ForwardMode.IBP:
        u_c, l_c = inputs
        return [K.log(u_c), K.log(l_c)]
    elif mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
        if mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs
            u_c = get_upper(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
            l_c = get_lower(x_0, w_l, b_l, perturbation_domain=perturbation_domain)
            l_c = K.maximum(K.cast(K.epsilon(), dtype=l_c.dtype), l_c)
        else:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs

        u_c_out = K.log(u_c)
        l_c_out = K.log(l_c)

        y = (u_c + l_c) / 2.0

        w_l_0 = (u_c_out - l_c_out) / K.maximum(u_c - l_c, K.epsilon())
        b_l_0 = l_c_out - w_l_0 * l_c

        w_u_0 = 1 / y
        b_u_0 = K.log(y) - 1

        w_u_out = w_u_0[:, None] * w_u
        b_u_out = w_u_0 * b_u + b_u_0
        w_l_out = w_l_0[:, None] * w_l
        b_l_out = w_l_0 * b_l + b_l_0

        if mode == ForwardMode.AFFINE:
            return [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
        else:
            return [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")


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
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    if mode == ForwardMode.IBP:
        return [K.exp(e) for e in inputs]
    elif mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
        if mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs
            u_c = get_upper(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
            l_c = get_lower(x_0, w_l, b_l, perturbation_domain=perturbation_domain)
        else:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs

        u_c_out = K.exp(u_c)
        l_c_out = K.exp(l_c)

        y = (u_c + l_c) / 2.0  # do finetuneting

        w_u_0 = (u_c_out - l_c_out) / K.maximum(u_c - l_c, K.epsilon())
        b_u_0 = l_c_out - w_u_0 * l_c

        w_l_0 = K.exp(y)
        b_l_0 = w_l_0 * (1 - y)

        w_u_out = w_u_0[:, None] * w_u
        b_u_out = w_u_0 * b_u + b_u_0
        w_l_out = w_l_0[:, None] * w_l
        b_l_out = w_l_0 * b_l + b_l_0

        if mode == ForwardMode.AFFINE:
            return [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
        else:
            return [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")
