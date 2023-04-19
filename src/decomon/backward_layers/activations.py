import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

from decomon.backward_layers.utils import backward_relu_, backward_softplus_
from decomon.layers.core import ForwardMode, StaticVariables
from decomon.utils import (
    Slope,
    get_linear_hull_relu,
    get_linear_hull_s_shape,
    get_linear_hull_sigmoid,
    get_linear_hull_tanh,
    get_linear_softplus_hull,
    get_lower,
    get_upper,
    sigmoid_prime,
    softsign_prime,
    tanh_prime,
)

ELU = "elu"
SELU = "selu"
SOFTPLUS = "softplus"
SOFTSIGN = "softsign"
SOFTMAX = "softmax"
RELU = "relu_"
RELU_ = "relu"
SIGMOID = "sigmoid"
TANH = "tanh"
EXPONENTIAL = "exponential"
HARD_SIGMOID = "hard_sigmoid"
LINEAR = "linear"


def backward_relu(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    alpha: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward  LiRPA of relu

    Args:
        x
        dc_decomp
        convex_domain
        alpha
        max_value
        threshold
        slope
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    mode = ForwardMode(mode)
    if dc_decomp:
        raise NotImplementedError()

    if threshold != 0:
        raise NotImplementedError()

    if not alpha and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
        if mode == ForwardMode.IBP:
            upper, lower = x[:nb_tensors]
        elif mode == ForwardMode.AFFINE:
            z_, w_u_, b_u_, w_l_, b_l_ = x[:nb_tensors]
            upper = get_upper(z_, w_u_, b_u_, convex_domain=convex_domain)
            lower = get_lower(z_, w_l_, b_l_, convex_domain=convex_domain)
        elif mode == ForwardMode.HYBRID:
            _, upper, _, _, lower, _, _ = x[:nb_tensors]
        else:
            raise ValueError(f"Unknown mode {mode}")
        bounds = get_linear_hull_relu(upper, lower, slope=slope, **kwargs)
        shape = np.prod(x[-1].shape[1:])
        bounds = [K.reshape(elem, (-1, shape)) for elem in bounds]

        w_u_, b_u_, w_l_, b_l_ = bounds
        return [w_u_, b_u_, w_l_, b_l_]

    raise NotImplementedError()


def backward_sigmoid(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward  LiRPA of sigmoid

    Args:
        inputs
        dc_decomp
        convex_domain
        slope
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == ForwardMode.IBP:
        upper, lower = inputs[:nb_tensors]
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = inputs[:nb_tensors]
        upper = get_upper(z_, w_u_, b_u_)
        lower = get_lower(z_, w_l_, b_l_)
    elif mode == ForwardMode.HYBRID:
        _, upper, _, _, lower, _, _ = inputs[:nb_tensors]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return get_linear_hull_sigmoid(upper, lower, slope=slope, **kwargs)


def backward_tanh(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward  LiRPA of tanh

    Args:
        inputs
        dc_decomp
        convex_domain
        slope
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == ForwardMode.IBP:
        upper, lower = inputs[:nb_tensors]
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = inputs[:nb_tensors]
        upper = get_upper(z_, w_u_, b_u_)
        lower = get_lower(z_, w_l_, b_l_)
    elif mode == ForwardMode.HYBRID:
        _, upper, _, _, lower, _, _ = inputs[:nb_tensors]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return get_linear_hull_tanh(upper, lower, slope=slope, **kwargs)


def backward_hard_sigmoid(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward  LiRPA of hard sigmoid

    Args:
        x
        dc_decomp
        convex_domain
        slope
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    raise NotImplementedError()


def backward_elu(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward  LiRPA of Exponential Linear Unit

    Args:
        x
        dc_decomp
        convex_domain
        slope
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def backward_selu(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward LiRPA of Scaled Exponential Linear Unit (SELU)

    Args:
        x
        dc_decomp
        convex_domain
        slope
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def backward_linear(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward LiRPA of linear

    Args:
        x
        dc_decomp
        convex_domain
        slope
        mode

    Returns:

    """
    if convex_domain is None:
        convex_domain = {}
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def backward_exponential(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward LiRPAof exponential

    Args:
        x
        dc_decomp
        convex_domain
        slope
        mode

    Returns:

    """
    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def backward_softplus(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward LiRPA of softplus

    Args:
        x
        dc_decomp
        convex_domain
        slope
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == ForwardMode.IBP:
        upper, lower = x[:nb_tensors]
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = x[:nb_tensors]
        upper = get_upper(z_, w_u_, b_u_)
        lower = get_lower(z_, w_l_, b_l_)
    elif mode == ForwardMode.HYBRID:
        _, upper, _, _, lower, _, _ = x[:nb_tensors]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return get_linear_softplus_hull(upper, lower, slope=slope, **kwargs)


def backward_softsign(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward LiRPA of softsign

    Args:
        x
        w_out_u
        b_out_u
        w_out_l
        b_out_l
        convex_domain
        slope: backward slope
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}

    bounds = get_linear_hull_s_shape(x, func=K.softsign, f_prime=softsign_prime, convex_domain=convex_domain, mode=mode)
    shape = np.prod(x[-1].shape[1:])
    bounds = [K.reshape(elem, (-1, shape)) for elem in bounds]

    w_u_, b_u_, w_l_, b_l_ = bounds
    return [w_u_, b_u_, w_l_, b_l_]


def backward_softsign_(
    y: List[tf.Tensor],
    w_out_u: tf.Tensor,
    b_out_u: tf.Tensor,
    w_out_l: tf.Tensor,
    b_out_l: tf.Tensor,
    convex_domain: Optional[Dict[str, Any]] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[tf.Tensor]:

    if convex_domain is None:
        convex_domain = {}
    w_u_0, b_u_0, w_l_0, b_l_0 = get_linear_hull_s_shape(
        y, func=K.softsign, f_prime=softsign_prime, convex_domain=convex_domain, mode=mode, slope=slope
    )

    w_u_0 = K.expand_dims(w_u_0, -1)
    w_l_0 = K.expand_dims(w_l_0, -1)

    b_u_0 = K.expand_dims(b_u_0, -1)
    b_l_0 = K.expand_dims(b_l_0, -1)

    z_value = K.cast(0.0, dtype=y[0].dtype)
    w_out_u_ = K.maximum(z_value, w_out_u) * w_u_0 + K.minimum(z_value, w_out_u) * w_l_0
    w_out_l_ = K.maximum(z_value, w_out_l) * w_l_0 + K.minimum(z_value, w_out_l) * w_u_0
    b_out_u_ = K.sum(K.maximum(z_value, w_out_u) * b_u_0 + K.minimum(z_value, w_out_u) * b_l_0, 2) + b_out_u
    b_out_l_ = K.sum(K.maximum(z_value, w_out_l) * b_l_0 + K.minimum(z_value, w_out_l) * b_u_0, 2) + b_out_l

    return [w_out_u_, b_out_u_, w_out_l_, b_out_l_]


def backward_softmax(
    x: List[tf.Tensor],
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward LiRPA of softmax

    Args:
        x
        dc_decomp
        convex_domain
        slope
        mode
        axis

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def deserialize(name: str) -> Callable[..., List[tf.Tensor]]:
    """Get the activation from name.

    Args:
        name: name of the method.
    among the implemented Keras activation function.

    Returns:

    """
    name = name.lower()

    if name == SOFTMAX:
        return backward_softmax
    if name == ELU:
        return backward_elu
    if name == SELU:
        return backward_selu
    if name == SOFTPLUS:
        return backward_softplus
    if name == SOFTSIGN:
        return backward_softsign
    if name == SIGMOID:
        return backward_sigmoid
    if name == TANH:
        return backward_tanh
    if name in [RELU, RELU_]:
        return backward_relu
    if name == EXPONENTIAL:
        return backward_exponential
    if name == LINEAR:
        return backward_linear
    raise ValueError("Could not interpret " "activation function identifier:", name)


def get(identifier: Any) -> Callable[..., List[tf.Tensor]]:
    """Get the `identifier` activation function.

    Args:
        identifier: None or str, name of the function.

    Returns:
        The activation function, `linear` if `identifier` is None.

    """
    if identifier is None:
        return backward_linear
    if isinstance(identifier, str):
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn(
                "Do not pass a layer instance (such as {identifier}) as the "
                "activation argument of another layer. Instead, advanced "
                "activation layers should be used just like any other "
                "layer in a model.".format(identifier=identifier.__class__.__name__)
            )
        return identifier
    else:
        raise ValueError("Could not interpret " "activation function identifier:", identifier)
