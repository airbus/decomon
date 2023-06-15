import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.types.experimental import TensorLike

from decomon.core import BoxDomain, InputsOutputsSpec, PerturbationDomain, Slope
from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.utils import exp, expand_dims, frac_pos, multiply, softplus_, sum
from decomon.utils import (
    get_linear_hull_s_shape,
    get_lower,
    get_upper,
    minus,
    relu_,
    sigmoid_prime,
    softsign_prime,
    tanh_prime,
)

ELU = "elu"
SELU = "selu"
SOFTPLUS = "softplus"
SOFTSIGN = "softsign"
SOFTMAX = "softmax"
RELU = "relu"
SIGMOID = "sigmoid"
TANH = "tanh"
EXPONENTIAL = "exponential"
HARD_SIGMOID = "hard_sigmoid"
LINEAR = "linear"
GROUP_SORT_2 = "GroupSort2"


def relu(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    alpha: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """
    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        alpha: see Keras official documentation
        max_value: see Keras official documentation
        threshold: see Keras official documentation
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if threshold != 0:
        raise NotImplementedError()

    if not alpha and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        return relu_(
            inputs, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, slope=slope, **kwargs
        )

    raise NotImplementedError()


def linear_hull_s_shape(
    inputs: List[tf.Tensor],
    func: Callable[[TensorLike], tf.Tensor] = K.sigmoid,
    f_prime: Callable[[TensorLike], tf.Tensor] = sigmoid_prime,
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
) -> List[tf.Tensor]:
    """Computing the linear hull of s-shape functions

    Args:
        inputs: list of input tensors
        func: the function (sigmoid, tanh, softsign...)
        f_prime: the derivative of the function (sigmoid_prime...)
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
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
        u_c, l_c = inputs[: InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode).nb_tensors]
    elif mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode).nb_tensors]
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs[: InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode).nb_tensors]
        u_c = get_upper(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
        l_c = get_lower(x_0, w_l, b_l, perturbation_domain=perturbation_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
        u_c_out = func(u_c)
        l_c_out = func(l_c)

    if mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:

        w_u_0, b_u_0, w_l_0, b_l_0 = get_linear_hull_s_shape(
            inputs, func=func, f_prime=f_prime, perturbation_domain=perturbation_domain, mode=mode
        )

        if len(w_u.shape) == len(b_u.shape):
            # it happens with the convert function to spare memory footprint
            n_dim = np.prod(w_u.shape[1:])
            M = np.reshape(
                np.diag([K.cast(1, dtype=w_u_0.dtype)] * n_dim), [1, n_dim] + list(w_u.shape[1:])
            )  # usage de numpy pb pour les types
            w_u_out = M * K.concatenate([K.expand_dims(w_u_0, 1)] * n_dim, 1)
            w_l_out = M * K.concatenate([K.expand_dims(w_l_0, 1)] * n_dim, 1)
            b_u_out = b_u_0
            b_l_out = b_l_0

        else:
            w_u_out = K.expand_dims(w_u_0, 1) * w_u  # pour l'instant
            b_u_out = b_u_0 + w_u_0 * b_u
            w_l_out = K.expand_dims(w_l_0, 1) * w_l
            b_l_out = b_l_0 + w_l_0 * b_l

    if mode == ForwardMode.IBP:
        output = [u_c_out, l_c_out]
    elif mode == ForwardMode.HYBRID:
        output = [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    elif mode == ForwardMode.AFFINE:
        output = [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
    else:
        raise ValueError(f"Unknown mode {mode}")

    return output


def sigmoid(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Sigmoid activation function .
    `1 / (1 + exp(-x))`.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    func = K.sigmoid
    f_prime = sigmoid_prime
    return linear_hull_s_shape(
        inputs, func, f_prime, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, slope=slope
    )


def tanh(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Hyperbolic activation function.
    `tanh(x)=2*sigmoid(2*x)+1`

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    func = K.tanh
    f_prime = tanh_prime
    return linear_hull_s_shape(
        inputs, func, f_prime, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, slope=slope
    )


def hard_sigmoid(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Hard sigmoid activation function.
       Faster to compute than sigmoid activation.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    raise NotImplementedError()


def elu(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Exponential linear unit.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    raise NotImplementedError()


def selu(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Scaled Exponential Linear Unit (SELU).

    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are predefined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    raise NotImplementedError()


def linear(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA foe Linear (i.e. identity) activation function.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    return inputs


def exponential(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Exponential activation function.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    return exp(inputs, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, slope=slope, **kwargs)


def softplus(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Softplus activation function `log(exp(x) + 1)`.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()

    return softplus_(inputs, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, slope=slope)


def softsign(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Softsign activation function `x / (abs(x) + 1)`.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    func = K.softsign
    f_prime = softsign_prime
    return linear_hull_s_shape(
        inputs, func, f_prime, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, slope=slope
    )


def softmax(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    clip: bool = True,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """LiRPA for Softmax activation function.

    Args:
        inputs: list of input tensors
        dc_decomp: boolean that indicates
        perturbation_domain: type of convex input domain (None or dict)
        mode: type of Forward propagation (ibp, affine, or hybrid)
        slope:
        **kwargs: see Keras official documentation
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)

    outputs_exp = exponential(
        minus(inputs, mode=mode), dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, slope=slope
    )
    outputs = expand_dims(
        frac_pos(
            sum(outputs_exp, axis=axis, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode),
            dc_decomp=dc_decomp,
            perturbation_domain=perturbation_domain,
            mode=mode,
        ),
        mode=mode,
        axis=axis,
    )
    outputs = multiply(outputs_exp, outputs, mode=mode, perturbation_domain=perturbation_domain)

    if mode == ForwardMode.IBP:
        u_c, l_c = outputs
        if clip:
            return [K.minimum(u_c, 1.0), K.maximum(l_c, 0.0)]
        else:
            return outputs
    if mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = outputs
        if clip:
            u_c = K.minimum(u_c, 1.0)
            l_c = K.maximum(l_c, 0.0)
        return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]

    return outputs


def group_sort_2(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    data_format: str = "channels_last",
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    raise NotImplementedError()


def deserialize(name: str) -> Callable[..., List[tf.Tensor]]:
    """Get the activation from name.

    Args:
        name: name of the method.
    among the implemented Keras activation function.

    Returns:
        the activation function
    """
    name = name.lower()

    if name == SOFTMAX:
        return softmax
    elif name == ELU:
        return elu
    elif name == SELU:
        return selu
    elif name == SOFTPLUS:
        return softplus
    elif name == SOFTSIGN:
        return softsign
    elif name == SIGMOID:
        return sigmoid
    elif name == TANH:
        return tanh
    elif name == RELU:
        return relu
    elif name == EXPONENTIAL:
        return exponential
    elif name == LINEAR:
        return linear
    elif name == GROUP_SORT_2:
        return group_sort_2
    else:
        raise ValueError(f"Could not interpret activation function identifier: {name}")


def get(identifier: Any) -> Callable[..., List[tf.Tensor]]:
    """Get the `identifier` activation function.

    Args:
        identifier: None or str, name of the function.

    Returns:
        The activation function, `linear` if `identifier` is None.

    """
    if identifier is None:
        return linear
    elif isinstance(identifier, str):
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, DecomonLayer):
            warnings.warn(
                "Do not pass a layer instance (such as {identifier}) as the "
                "activation argument of another layer. Instead, advanced "
                "activation layers should be used just like any other "
                "layer in a model.".format(identifier=identifier.__class__.__name__)
            )
        return identifier
    else:
        raise ValueError("Could not interpret " "activation function identifier:", identifier)
