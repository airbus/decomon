import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import keras_core as keras
import keras_core.backend as K
import numpy as np
from keras_core.layers import Layer

from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    Slope,
)
from decomon.utils import (
    get_linear_hull_relu,
    get_linear_hull_s_shape,
    get_linear_hull_sigmoid,
    get_linear_hull_tanh,
    get_linear_softplus_hull,
    sigmoid_prime,
    softsign_prime,
    tanh_prime,
)

ELU = "elu"
SELU = "selu"
SOFTPLUS = "softplus"
SOFTSIGN = "softsign"
SOFTMAX = "softmax"
RELU_ = "relu_"
RELU = "relu"
SIGMOID = "sigmoid"
TANH = "tanh"
EXPONENTIAL = "exponential"
HARD_SIGMOID = "hard_sigmoid"
LINEAR = "linear"


def backward_relu(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    alpha: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward  LiRPA of relu

    Args:
        inputs
        dc_decomp
        perturbation_domain
        alpha
        max_value
        threshold
        slope
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    if dc_decomp:
        raise NotImplementedError()

    if threshold != 0:
        raise NotImplementedError()

    if not alpha and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
        input_shape = inputs_outputs_spec.get_input_shape(inputs)
        bounds = get_linear_hull_relu(upper=u_c, lower=l_c, slope=slope, **kwargs)
        dim = np.prod(input_shape[1:])
        return [K.reshape(elem, (-1, dim)) for elem in bounds]

    raise NotImplementedError()


def backward_sigmoid(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward  LiRPA of sigmoid

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    return get_linear_hull_sigmoid(u_c, l_c, slope=slope, **kwargs)


def backward_tanh(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward  LiRPA of tanh

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    return get_linear_hull_tanh(u_c, l_c, slope=slope, **kwargs)


def backward_hard_sigmoid(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward  LiRPA of hard sigmoid

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    raise NotImplementedError()


def backward_elu(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward  LiRPA of Exponential Linear Unit

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def backward_selu(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward LiRPA of Scaled Exponential Linear Unit (SELU)

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def backward_linear(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward LiRPA of linear

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def backward_exponential(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward LiRPAof exponential

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def backward_softplus(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward LiRPA of softplus

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    return get_linear_softplus_hull(u_c, l_c, slope=slope, **kwargs)


def backward_softsign(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward LiRPA of softsign

    Args:
        inputs
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        slope: backward slope
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    bounds = get_linear_hull_s_shape(
        inputs, func=K.softsign, f_prime=softsign_prime, perturbation_domain=perturbation_domain, mode=mode
    )
    shape = np.prod(inputs[-1].shape[1:])
    return [K.reshape(elem, (-1, shape)) for elem in bounds]


def backward_softsign_(
    inputs: List[keras.KerasTensor],
    w_u_out: keras.KerasTensor,
    b_u_out: keras.KerasTensor,
    w_l_out: keras.KerasTensor,
    b_l_out: keras.KerasTensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    w_u_0, b_u_0, w_l_0, b_l_0 = get_linear_hull_s_shape(
        inputs, func=K.softsign, f_prime=softsign_prime, perturbation_domain=perturbation_domain, mode=mode, slope=slope
    )

    w_u_0 = K.expand_dims(w_u_0, -1)
    w_l_0 = K.expand_dims(w_l_0, -1)

    b_u_0 = K.expand_dims(b_u_0, -1)
    b_l_0 = K.expand_dims(b_l_0, -1)

    z_value = K.cast(0.0, dtype=inputs[0].dtype)
    b_u_out = K.sum(K.maximum(z_value, w_u_out) * b_u_0 + K.minimum(z_value, w_u_out) * b_l_0, 2) + b_u_out
    b_l_out = K.sum(K.maximum(z_value, w_l_out) * b_l_0 + K.minimum(z_value, w_l_out) * b_u_0, 2) + b_l_out
    w_u_out = K.maximum(z_value, w_u_out) * w_u_0 + K.minimum(z_value, w_u_out) * w_l_0
    w_l_out = K.maximum(z_value, w_l_out) * w_l_0 + K.minimum(z_value, w_l_out) * w_u_0

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def backward_softmax(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Backward LiRPA of softmax

    Args:
        inputs
        dc_decomp
        perturbation_domain
        slope
        mode
        axis

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if dc_decomp:
        raise NotImplementedError()
    mode = ForwardMode(mode)
    slope = Slope(slope)
    raise NotImplementedError()


def deserialize(name: str) -> Callable[..., List[keras.KerasTensor]]:
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


def get(identifier: Any) -> Callable[..., List[keras.KerasTensor]]:
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
