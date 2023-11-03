import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import keras
import keras.ops as K
import numpy as np

from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    Slope,
    get_affine,
    get_ibp,
)
from decomon.layers.core import DecomonLayer
from decomon.layers.utils import exp, expand_dims, frac_pos, multiply, softplus_, sum
from decomon.utils import (
    get_linear_hull_s_shape,
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    alpha: float = 0.0,
    max_value: Optional[float] = None,
    threshold: float = 0.0,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    func: Callable[[keras.KerasTensor], keras.KerasTensor] = K.sigmoid,
    f_prime: Callable[[keras.KerasTensor], keras.KerasTensor] = sigmoid_prime,
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
) -> List[keras.KerasTensor]:
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

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, compute_ibp_from_affine=False
    )
    dtype = x.dtype
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    if ibp:
        u_c_out = func(u_c)
        l_c_out = func(l_c)
    else:
        u_c_out, l_c_out = empty_tensor, empty_tensor

    if affine:
        w_u_0, b_u_0, w_l_0, b_l_0 = get_linear_hull_s_shape(
            inputs, func=func, f_prime=f_prime, perturbation_domain=perturbation_domain, mode=mode
        )
        if len(w_u.shape) == len(b_u.shape):
            # it happens with the convert function to spare memory footprint
            n_dim = int(np.prod(w_u.shape[1:]))
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
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError()
    else:
        h_out, g_out = empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def sigmoid(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    return inputs


def exponential(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    clip: bool = True,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
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
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    outputs_exp = exponential(
        minus(inputs, mode=mode, perturbation_domain=perturbation_domain),
        dc_decomp=dc_decomp,
        perturbation_domain=perturbation_domain,
        mode=mode,
        slope=slope,
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
        perturbation_domain=perturbation_domain,
    )
    outputs = multiply(outputs_exp, outputs, mode=mode, perturbation_domain=perturbation_domain)
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        outputs, compute_ibp_from_affine=False
    )
    if ibp:
        o_value = K.cast(1.0, dtype=u_c.dtype)
        z_value = K.cast(0.0, dtype=u_c.dtype)
        u_c = K.minimum(u_c, o_value)
        l_c = K.maximum(l_c, z_value)

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs([x, u_c, w_u, b_u, l_c, w_l, b_l, h, g])


def group_sort_2(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    data_format: str = "channels_last",
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    raise NotImplementedError()


def deserialize(name: str) -> Callable[..., List[keras.KerasTensor]]:
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


def get(identifier: Any) -> Callable[..., List[keras.KerasTensor]]:
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
