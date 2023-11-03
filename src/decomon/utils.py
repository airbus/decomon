from typing import Any, Callable, List, Optional, Tuple, Union

import keras
import keras.ops as K
import numpy as np
from keras.config import epsilon

from decomon.core import (
    BoxDomain,
    ForwardMode,
    GridDomain,
    InputsOutputsSpec,
    PerturbationDomain,
    Slope,
    get_affine,
    get_ibp,
)

TensorFunction = Callable[[keras.KerasTensor], keras.KerasTensor]


# linear hull for activation function
def relu_prime(x: keras.KerasTensor) -> keras.KerasTensor:
    """Derivative of relu

    Args:
        x

    Returns:

    """

    return K.clip(K.sign(x), K.cast(0, dtype=x.dtype), K.cast(1, dtype=x.dtype))


def sigmoid_prime(x: keras.KerasTensor) -> keras.KerasTensor:
    """Derivative of sigmoid

    Args:
        x

    Returns:

    """

    s_x = K.sigmoid(x)
    return s_x * (K.cast(1, dtype=x.dtype) - s_x)


def tanh_prime(x: keras.KerasTensor) -> keras.KerasTensor:
    """Derivative of tanh

    Args:
        x

    Returns:

    """

    s_x = K.tanh(x)
    return K.cast(1, dtype=x.dtype) - K.power(s_x, K.cast(2, dtype=x.dtype))


def softsign_prime(x: keras.KerasTensor) -> keras.KerasTensor:
    """Derivative of softsign

    Args:
        x

    Returns:

    """

    return K.cast(1.0, dtype=x.dtype) / K.power(K.cast(1.0, dtype=x.dtype) + K.abs(x), K.cast(2, dtype=x.dtype))


##### corners ######
def get_lower_bound_grid(x: keras.KerasTensor, W: keras.KerasTensor, b: keras.KerasTensor, n: int) -> keras.KerasTensor:
    A, B = convert_lower_search_2_subset_sum(x, W, b, n)
    return subset_sum_lower(A, B, repeat=n)


def get_upper_bound_grid(x: keras.KerasTensor, W: keras.KerasTensor, b: keras.KerasTensor, n: int) -> keras.KerasTensor:
    return -get_lower_bound_grid(x, -W, -b, n)


def get_bound_grid(
    x: keras.KerasTensor,
    W_u: keras.KerasTensor,
    b_u: keras.KerasTensor,
    W_l: keras.KerasTensor,
    b_l: keras.KerasTensor,
    n: int,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    upper = get_upper_bound_grid(x, W_u, b_u, n)
    lower = get_lower_bound_grid(x, W_l, b_l, n)

    return upper, lower


# convert max Wx +b s.t Wx+b<=0 into a subset-sum problem with positive values
def convert_lower_search_2_subset_sum(
    x: keras.KerasTensor, W: keras.KerasTensor, b: keras.KerasTensor, n: int
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    x_min = x[:, 0]
    x_max = x[:, 1]

    if len(W.shape) > 3:
        W = K.reshape(W, (-1, W.shape[1], int(np.prod(W.shape[2:]))))
        b = K.reshape(b, (-1, int(np.prod(b.shape[1:]))))

    const = BoxDomain().get_lower(x, W, b)

    weights = K.abs(W) * K.expand_dims((x_max - x_min) / n, -1)
    return weights, const


def subset_sum_lower(W: keras.KerasTensor, b: keras.KerasTensor, repeat: int = 1) -> keras.KerasTensor:
    B = K.sort(W, axis=1)
    C = K.repeat(B, repeats=repeat, axis=1)
    C_reduced = K.cumsum(C, axis=1)
    D = K.minimum(K.sign(K.expand_dims(-b, 1) - C_reduced) + 1, K.cast(1.0, dtype=b.dtype))

    score = K.minimum(K.sum(D * C, 1) + b, K.cast(0.0, dtype=b.dtype))
    return score


# define routines to get linear relaxations useful both for forward and backward
def get_linear_hull_relu(
    upper: keras.KerasTensor,
    lower: keras.KerasTensor,
    slope: Union[str, Slope],
    upper_g: float = 0.0,
    lower_g: float = 0.0,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    slope = Slope(slope)
    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    alpha = (K.relu(upper) - K.relu(lower)) / K.maximum(K.cast(epsilon(), dtype=upper.dtype), upper - lower)

    # scaling factor for the upper bound on the relu
    # see README

    w_u = alpha
    b_u = K.relu(lower) - alpha * lower
    z_value = K.cast(0.0, dtype=upper.dtype)
    o_value = K.cast(1.0, dtype=upper.dtype)

    if slope == Slope.V_SLOPE:
        # 1 if upper<=-lower else 0
        index_a = -K.clip(K.sign(upper + lower) - o_value, -o_value, z_value)

        # 1 if upper>-lower else 0
        index_b = o_value - index_a
        w_l = index_b
        b_l = z_value * b_u

    elif slope == Slope.A_SLOPE:
        w_l = K.clip(K.sign(w_u - 0.5), 0, 1)
        b_l = z_value * b_u

    elif slope == Slope.Z_SLOPE:
        w_l = z_value * w_u
        b_l = z_value * b_u

    elif slope == Slope.O_SLOPE:
        w_l = z_value * w_u + o_value
        b_l = z_value * b_u

    elif slope == Slope.S_SLOPE:
        w_l = w_u
        b_l = z_value * b_u

    else:
        raise NotImplementedError(f"Not implemented for slope {slope}")

    if "upper_grid" in kwargs:
        raise NotImplementedError()

    gamma = o_value
    if "finetune" in kwargs:
        # retrieve variables to optimize the slopes
        gamma = kwargs["finetune"][None]

    w_l = gamma * w_l + (o_value - gamma) * (o_value - w_l)

    # check inactive relu state: u<=0
    index_dead = -K.clip(K.sign(upper) - o_value, -o_value, z_value)  # =1 if inactive state
    index_linear = K.clip(K.sign(lower) + o_value, z_value, o_value)  # 1 if linear state

    w_u = (o_value - index_dead) * w_u
    w_l = (o_value - index_dead) * w_l
    b_u = (o_value - index_dead) * b_u
    b_l = (o_value - index_dead) * b_l

    w_u = (o_value - index_linear) * w_u + index_linear
    w_l = (o_value - index_linear) * w_l + index_linear
    b_u = (o_value - index_linear) * b_u
    b_l = (o_value - index_linear) * b_l

    return [w_u, b_u, w_l, b_l]


def get_linear_hull_sigmoid(
    upper: keras.KerasTensor, lower: keras.KerasTensor, slope: Union[str, Slope], **kwargs: Any
) -> List[keras.KerasTensor]:
    x = [upper, lower]
    return get_linear_hull_s_shape(x, func=K.sigmoid, f_prime=sigmoid_prime, mode=ForwardMode.IBP, **kwargs)


def get_linear_hull_tanh(
    upper: keras.KerasTensor, lower: keras.KerasTensor, slope: Union[str, Slope], **kwargs: Any
) -> List[keras.KerasTensor]:
    x = [upper, lower]
    return get_linear_hull_s_shape(x, func=K.tanh, f_prime=tanh_prime, mode=ForwardMode.IBP, **kwargs)


def get_linear_softplus_hull(
    upper: keras.KerasTensor, lower: keras.KerasTensor, slope: Union[str, Slope], **kwargs: Any
) -> List[keras.KerasTensor]:
    slope = Slope(slope)
    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    u_c = K.softsign(upper)
    l_c = K.softsign(lower)
    alpha = (u_c - l_c) / K.maximum(K.cast(epsilon(), dtype=upper.dtype), (upper - lower))
    w_u = alpha
    b_u = -alpha * lower + l_c

    z_value = K.cast(0.0, dtype=upper.dtype)
    o_value = K.cast(1.0, dtype=upper.dtype)

    if slope == Slope.V_SLOPE:
        # 1 if upper<=-lower else 0
        index_a = -K.clip(K.sign(upper + lower) - o_value, -o_value, z_value)
        # 1 if upper>-lower else 0
        index_b = o_value - index_a
        w_l = index_b
        b_l = z_value * b_u
    elif slope == Slope.Z_SLOPE:
        w_l = z_value * w_u
        b_l = z_value * b_u
    elif slope == Slope.O_SLOPE:
        w_l = z_value * w_u + o_value
        b_l = z_value * b_u
    elif slope == Slope.S_SLOPE:
        w_l = w_u
        b_l = z_value * b_u
    else:
        raise ValueError(f"Unknown slope {slope}")

    index_dead = -K.clip(K.sign(upper) - o_value, -o_value, z_value)

    w_u = (o_value - index_dead) * w_u
    w_l = (o_value - index_dead) * w_l
    b_u = (o_value - index_dead) * b_u
    b_l = (o_value - index_dead) * b_l

    if "finetune" in kwargs:
        # weighted linear combination
        alpha_u, alpha_l = kwargs["finetune"]
        alpha_u = alpha_u[None]
        alpha_l = alpha_l[None]

        w_u = alpha_u * w_u
        b_u = alpha_u * b_u + (o_value - alpha_u) * K.maximum(upper, z_value)

        w_l = alpha_l * w_l
        b_l = alpha_l * b_l + (o_value - alpha_l) * K.maximum(lower, z_value)

    return [w_u, b_u, w_l, b_l]


def subtract(
    inputs_0: List[keras.KerasTensor],
    inputs_1: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[keras.KerasTensor]:
    """LiRPA implementation of inputs_0-inputs_1

    Args:
        inputs_0: tensor
        inputs_1: tensor
        dc_decomp: boolean that indicates
        perturbation_domain: the type of perturbation domain
    whether we return a difference of convex decomposition of our layer

    Returns:
        inputs_0 - inputs_1
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    inputs_1 = minus(inputs_1, mode=mode, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain)
    output = add(inputs_0, inputs_1, dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    return output


def add(
    inputs_0: List[keras.KerasTensor],
    inputs_1: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[keras.KerasTensor]:
    """LiRPA implementation of inputs_0+inputs_1

    Args:
        inputs_0: tensor
        inputs_1: tensor
        dc_decomp: boolean that indicates
        perturbation_domain: the type of perturbation domain
    whether we return a difference of convex decomposition of our layer

    Returns:
        inputs_0 + inputs_1
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, h_0, g_0 = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs_0
    )
    _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1, h_1, g_1 = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs_1
    )

    x = x_0
    h = h_0 + h_1
    g = g_0 + g_1
    u_c = u_c_0 + u_c_1
    l_c = l_c_0 + l_c_1
    w_u = w_u_0 + w_u_1
    w_l = w_l_0 + w_l_1
    b_u = b_u_0 + b_u_1
    b_l = b_l_0 + b_l_1

    fulloutputs = [x, u_c, w_u, b_u, l_c, w_l, b_l, h, g]
    outputs = inputs_outputs_spec.extract_outputsformode_from_fulloutputs(fulloutputs)

    return outputs


def relu_(
    inputs: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    dtype = x.dtype

    z_value = K.cast(0.0, dtype=dtype)
    o_value = K.cast(1.0, dtype=dtype)
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

    if dc_decomp:
        h_out = K.maximum(h, -g)
        g_out = g
        index_dead = -K.clip(K.sign(u_c) - o_value, -o_value, z_value)  # =1 if inactive state
        index_linear = K.clip(K.sign(l_c) + o_value, z_value, o_value)  # 1 if linear state

        h_out = (o_value - index_dead) * h_out
        g_out = (o_value - index_dead) * g_out
        h_out = (o_value - index_linear) * h_out + index_linear * h
        g_out = (o_value - index_linear) * g_out + index_linear * g
    else:
        h_out, g_out = empty_tensor, empty_tensor

    if ibp:
        u_c_out = K.relu(u_c)
        l_c_out = K.relu(l_c)
    else:
        u_c_out = empty_tensor
        l_c_out = empty_tensor

    if affine:
        if isinstance(perturbation_domain, GridDomain):
            upper_g, lower_g = get_bound_grid(x, w_u, b_u, w_l, b_l, 1)
            kwargs.update({"upper_grid": upper_g, "lower_grid": lower_g})

        w_u_out, b_u_out, w_l_out, b_l_out = get_linear_hull_relu(u_c, l_c, slope, **kwargs)
        b_u_out = w_u_out * b_u + b_u_out
        b_l_out = w_l_out * b_l + b_l_out
        w_u_out = K.expand_dims(w_u_out, 1) * w_u
        w_l_out = K.expand_dims(w_l_out, 1) * w_l
    else:
        w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def minus(
    inputs: List[keras.KerasTensor],
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """LiRPA implementation of minus(x)=-x.

    Args:
        inputs
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, tight=False, compute_ibp_from_affine=False
    )

    u_c_out = -l_c
    l_c_out = -u_c
    w_u_out = -w_l
    b_u_out = -b_l
    w_l_out = -w_u
    b_l_out = -b_u
    h_out = -g
    g_out = -h

    return inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
    )


def maximum(
    inputs_0: List[keras.KerasTensor],
    inputs_1: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    finetune: bool = False,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """LiRPA implementation of element-wise max

    Args:
        inputs_0: list of tensors
        inputs_1: list of tensors
        dc_decomp: boolean that indicates
        perturbation_domain: the type of perturbation domain
    whether we return a difference of convex decomposition of our layer

    Returns:
        maximum(inputs_0, inputs_1)
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    output_0 = subtract(inputs_1, inputs_0, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode)
    if finetune:
        finetune = kwargs["finetune_params"]
        output_1 = relu_(
            output_0, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, finetune=finetune
        )
    else:
        output_1 = relu_(output_0, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode)

    return add(
        output_1,
        inputs_0,
        dc_decomp=dc_decomp,
        perturbation_domain=perturbation_domain,
        mode=mode,
    )


def minimum(
    inputs_0: List[keras.KerasTensor],
    inputs_1: List[keras.KerasTensor],
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    finetune: bool = False,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """LiRPA implementation of element-wise min

    Args:
        inputs_0
        inputs_1
        dc_decomp
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    return minus(
        maximum(
            minus(inputs_0, dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain),
            minus(inputs_1, dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain),
            dc_decomp=dc_decomp,
            perturbation_domain=perturbation_domain,
            mode=mode,
            finetune=finetune,
            **kwargs,
        ),
        dc_decomp=dc_decomp,
        mode=mode,
        perturbation_domain=perturbation_domain,
    )


def get_linear_hull_s_shape(
    inputs: List[keras.KerasTensor],
    func: TensorFunction = K.sigmoid,
    f_prime: TensorFunction = sigmoid_prime,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    dc_decomp: bool = False,
    **kwargs: Any,
) -> List[keras.KerasTensor]:
    """Computing the linear hull of shape functions  given the pre activation neurons

    Args:
        inputs: list of input tensors
        func: the function (sigmoid, tanh, softsign...)
        f_prime: the derivative of the function (sigmoid_prime...)
        perturbation_domain: the type of convex input domain
        mode: type of Forward propagation (ibp, affine, or hybrid)

    Returns:
        the updated list of tensors
    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)

    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
    dtype = x.dtype

    z_value = K.cast(0.0, dtype=dtype)
    o_value = K.cast(1.0, dtype=dtype)
    t_value = K.cast(2.0, dtype=dtype)

    # flatten
    shape = list(u_c.shape[1:])
    u_c_flat = K.reshape(u_c, (-1, int(np.prod(shape))))  # (None, n)
    l_c_flat = K.reshape(l_c, (-1, int(np.prod(shape))))  # (None, n)

    # upper bound
    # derivative
    s_u_prime = f_prime(u_c_flat)  # (None, n)
    s_l_prime = f_prime(l_c_flat)  # (None, n)
    s_u = func(u_c_flat)  # (None, n)
    s_l = func(l_c_flat)  # (None, n)

    # case 0:
    coeff = (s_u - s_l) / K.maximum(K.cast(epsilon(), dtype=dtype), u_c_flat - l_c_flat)
    alpha_u_0 = K.where(
        K.greater_equal(s_u_prime, coeff), o_value + z_value * u_c_flat, z_value * u_c_flat
    )  # (None, n)
    alpha_u_1 = (o_value - alpha_u_0) * ((K.sign(l_c_flat) + o_value) / t_value)

    w_u_0 = coeff
    b_u_0 = -w_u_0 * l_c_flat + s_l

    w_u_1 = z_value * u_c_flat
    b_u_1 = s_u

    w_u_2, b_u_2 = get_t_upper(u_c_flat, l_c_flat, s_l, func=func, f_prime=f_prime)

    w_u_out = K.reshape(alpha_u_0 * w_u_0 + alpha_u_1 * w_u_1 + (o_value - alpha_u_0 - alpha_u_1) * w_u_2, [-1] + shape)
    b_u_out = K.reshape(alpha_u_0 * b_u_0 + alpha_u_1 * b_u_1 + (o_value - alpha_u_0 - alpha_u_1) * b_u_2, [-1] + shape)

    # linear hull
    # case 0:
    alpha_l_0 = K.where(
        K.greater_equal(s_l_prime, coeff), o_value + z_value * l_c_flat, z_value * l_c_flat
    )  # (None, n)
    alpha_l_1 = (o_value - alpha_l_0) * ((K.sign(-u_c_flat) + o_value) / t_value)

    w_l_0 = coeff
    b_l_0 = -w_l_0 * u_c_flat + s_u

    w_l_1 = z_value * u_c_flat
    b_l_1 = s_l

    w_l_2, b_l_2 = get_t_lower(u_c_flat, l_c_flat, s_u, func=func, f_prime=f_prime)

    w_l_out = K.reshape(alpha_l_0 * w_l_0 + alpha_l_1 * w_l_1 + (o_value - alpha_l_0 - alpha_l_1) * w_l_2, [-1] + shape)
    b_l_out = K.reshape(alpha_l_0 * b_l_0 + alpha_l_1 * b_l_1 + (o_value - alpha_l_0 - alpha_l_1) * b_l_2, [-1] + shape)

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def get_t_upper(
    u_c_flat: keras.KerasTensor,
    l_c_flat: keras.KerasTensor,
    s_l: keras.KerasTensor,
    func: TensorFunction = K.sigmoid,
    f_prime: TensorFunction = sigmoid_prime,
) -> List[keras.KerasTensor]:
    """linear interpolation between lower and upper bounds on the function func to have a symbolic approximation of the best
    coefficient for the affine upper bound

    Args:
        u_c_flat: flatten tensor of constant upper bound
        l_c_flat: flatten tensor of constant lower bound
        s_l: lowest value of the function func on the domain
        func: the function (sigmoid, tanh,  softsign)
        f_prime: the derivative of the function

    Returns:
        the upper affine bounds in this subcase
    """

    o_value = K.cast(1.0, dtype=u_c_flat.dtype)
    z_value = K.cast(0.0, dtype=u_c_flat.dtype)

    # step1: find t
    u_c_reshaped = K.expand_dims(u_c_flat, -1)  # (None, n , 1)
    l_c_reshaped = K.expand_dims(l_c_flat, -1)  # (None, n,  1)
    t = K.cast(np.linspace(0, 1, 100)[None, None, :], dtype=u_c_flat.dtype) * u_c_reshaped  # (None, n , 100)

    s_p_t = f_prime(t)  # (None, n, 100)
    s_t = func(t)  # (None, n, 100)

    score = K.abs(s_p_t - (s_t - K.expand_dims(s_l, -1)) / (t - l_c_reshaped))  # (None, n, 100)
    index = K.argmin(score, -1)  # (None, n)
    threshold = K.min(score, -1)  # (None, n)

    index_t = K.cast(
        K.where(K.greater(threshold, z_value * threshold), index, K.clip(index - 1, 0, 100)), dtype=u_c_flat.dtype
    )  # (None, n)
    t_value = K.sum(
        K.where(
            K.equal(
                o_value * K.cast(np.arange(0, 100)[None, None, :], dtype=u_c_flat.dtype) + z_value * u_c_reshaped,
                K.expand_dims(index_t, -1) + z_value * u_c_reshaped,
            ),
            t,
            z_value * t,
        ),
        -1,
    )  # (None, n)

    s_t = func(t_value)  # (None, n)
    w_u = (s_t - s_l) / K.maximum(K.cast(epsilon(), dtype=u_c_flat.dtype), t_value - l_c_flat)  # (None, n)
    b_u = -w_u * l_c_flat + s_l  # + func(l_c_flat)

    return [w_u, b_u]


def get_t_lower(
    u_c_flat: keras.KerasTensor,
    l_c_flat: keras.KerasTensor,
    s_u: keras.KerasTensor,
    func: TensorFunction = K.sigmoid,
    f_prime: TensorFunction = sigmoid_prime,
) -> List[keras.KerasTensor]:
    """linear interpolation between lower and upper bounds on the function func to have a symbolic approximation of the best
    coefficient for the affine lower bound

    Args:
        u_c_flat: flatten tensor of constant upper bound
        l_c_flat: flatten tensor of constant lower bound
        s_u: highest value of the function func on the domain
        func: the function (sigmoid, tanh,  softsign)
        f_prime: the derivative of the function

    Returns:
        the lower affine bounds in this subcase
    """
    z_value = K.cast(0.0, dtype=u_c_flat.dtype)
    o_value = K.cast(1.0, dtype=u_c_flat.dtype)

    # step1: find t
    u_c_reshaped = K.expand_dims(u_c_flat, -1)  # (None, n , 1)
    l_c_reshaped = K.expand_dims(l_c_flat, -1)  # (None, n,  1)
    t = K.cast(np.linspace(0, 1.0, 100)[None, None, :], dtype=u_c_flat.dtype) * l_c_reshaped  # (None, n , 100)

    s_p_t = f_prime(t)  # (None, n, 100)
    s_t = func(t)  # (None, n, 100)

    score = K.abs(s_p_t - (K.expand_dims(s_u, -1) - s_t) / (u_c_reshaped - t))  # (None, n, 100)
    index = K.argmin(score, -1)  # (None, n)

    threshold = K.min(score, -1)
    index_t = K.cast(
        K.where(K.greater(threshold, z_value * threshold), index, K.clip(index + 1, 0, 100)), dtype=u_c_flat.dtype
    )  # (None, n)
    t_value = K.sum(
        K.where(
            K.equal(
                o_value * K.cast(np.arange(0, 100)[None, None, :], dtype=u_c_flat.dtype) + z_value * u_c_reshaped,
                K.expand_dims(index_t, -1) + z_value * u_c_reshaped,
            ),
            t,
            z_value * t,
        ),
        -1,
    )

    s_t = func(t_value)  # (None, n)
    w_l = (s_u - s_t) / K.maximum(K.cast(epsilon(), dtype=u_c_flat.dtype), u_c_flat - t_value)  # (None, n)
    b_l = -w_l * u_c_flat + s_u  # func(u_c_flat)

    return [w_l, b_l]
