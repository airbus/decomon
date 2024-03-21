from collections.abc import Callable
from typing import Any, Union

import numpy as np
from keras import ops as K
from keras.src.backend import epsilon

from decomon.constants import Slope
from decomon.types import Tensor

TensorFunction = Callable[[Tensor], Tensor]


def sigmoid_prime(x: Tensor) -> Tensor:
    """Derivative of sigmoid

    Args:
        x

    Returns:

    """

    s_x = K.sigmoid(x)
    return s_x * (K.cast(1, dtype=x.dtype) - s_x)


def tanh_prime(x: Tensor) -> Tensor:
    """Derivative of tanh

    Args:
        x

    Returns:

    """

    s_x = K.tanh(x)
    return K.cast(1, dtype=x.dtype) - K.power(s_x, K.cast(2, dtype=x.dtype))


def get_linear_hull_relu(
    upper: Tensor,
    lower: Tensor,
    slope: Union[str, Slope],
    upper_g: float = 0.0,
    lower_g: float = 0.0,
    **kwargs: Any,
) -> list[Tensor]:
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


def get_linear_softplus_hull(upper: Tensor, lower: Tensor, slope: Union[str, Slope], **kwargs: Any) -> list[Tensor]:
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


def relu_prime(x: Tensor) -> Tensor:
    """Derivative of relu

    Args:
        x

    Returns:

    """

    return K.clip(K.sign(x), K.cast(0, dtype=x.dtype), K.cast(1, dtype=x.dtype))


def softsign_prime(x: Tensor) -> Tensor:
    """Derivative of softsign

    Args:
        x

    Returns:

    """

    return K.cast(1.0, dtype=x.dtype) / K.power(K.cast(1.0, dtype=x.dtype) + K.abs(x), K.cast(2, dtype=x.dtype))


def get_linear_hull_s_shape(
    lower: Tensor,
    upper: Tensor,
    func: TensorFunction = K.sigmoid,
    f_prime: TensorFunction = sigmoid_prime,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computing the linear hull of shape functions  given the pre activation neurons

    Args:
        lower: lower bound on keras input
        upper: upper bound on keras input
        func: the function (sigmoid, tanh, softsign...)
        f_prime: the derivative of the function (sigmoid_prime...)
        perturbation_domain: the type of convex input domain
        mode: type of Forward propagation (ibp, affine, or hybrid)

    Returns:
        w_l, b_l, w_u, b_u: affine bounds on activation layer
    """

    dtype = lower.dtype

    z_value = K.cast(0.0, dtype=dtype)
    o_value = K.cast(1.0, dtype=dtype)
    t_value = K.cast(2.0, dtype=dtype)

    # flatten
    shape = list(lower.shape[1:])
    upper_flat = K.reshape(upper, (-1, int(np.prod(shape))))  # (None, n)
    lower_flat = K.reshape(lower, (-1, int(np.prod(shape))))  # (None, n)

    # upper bound
    # derivative
    s_u_prime = f_prime(upper_flat)  # (None, n)
    s_l_prime = f_prime(lower_flat)  # (None, n)
    s_u = func(upper_flat)  # (None, n)
    s_l = func(lower_flat)  # (None, n)

    # case 0:
    coeff = (s_u - s_l) / K.maximum(K.cast(epsilon(), dtype=dtype), upper_flat - lower_flat)
    alpha_u_0 = K.where(
        K.greater_equal(s_u_prime, coeff), o_value + z_value * upper_flat, z_value * upper_flat
    )  # (None, n)
    alpha_u_1 = (o_value - alpha_u_0) * ((K.sign(lower_flat) + o_value) / t_value)

    w_u_0 = coeff
    b_u_0 = -w_u_0 * lower_flat + s_l

    w_u_1 = z_value * upper_flat
    b_u_1 = s_u

    w_u_2, b_u_2 = get_t_upper(upper_flat, lower_flat, s_l, func=func, f_prime=f_prime)

    w_u_out = K.reshape(alpha_u_0 * w_u_0 + alpha_u_1 * w_u_1 + (o_value - alpha_u_0 - alpha_u_1) * w_u_2, [-1] + shape)
    b_u_out = K.reshape(alpha_u_0 * b_u_0 + alpha_u_1 * b_u_1 + (o_value - alpha_u_0 - alpha_u_1) * b_u_2, [-1] + shape)

    # linear hull
    # case 0:
    alpha_l_0 = K.where(
        K.greater_equal(s_l_prime, coeff), o_value + z_value * lower_flat, z_value * lower_flat
    )  # (None, n)
    alpha_l_1 = (o_value - alpha_l_0) * ((K.sign(-upper_flat) + o_value) / t_value)

    w_l_0 = coeff
    b_l_0 = -w_l_0 * upper_flat + s_u

    w_l_1 = z_value * upper_flat
    b_l_1 = s_l

    w_l_2, b_l_2 = get_t_lower(upper_flat, lower_flat, s_u, func=func, f_prime=f_prime)

    w_l_out = K.reshape(alpha_l_0 * w_l_0 + alpha_l_1 * w_l_1 + (o_value - alpha_l_0 - alpha_l_1) * w_l_2, [-1] + shape)
    b_l_out = K.reshape(alpha_l_0 * b_l_0 + alpha_l_1 * b_l_1 + (o_value - alpha_l_0 - alpha_l_1) * b_l_2, [-1] + shape)

    return w_l_out, b_l_out, w_u_out, b_u_out


def get_t_upper(
    u_c_flat: Tensor,
    l_c_flat: Tensor,
    s_l: Tensor,
    func: TensorFunction = K.sigmoid,
    f_prime: TensorFunction = sigmoid_prime,
) -> tuple[Tensor, Tensor]:
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

    return w_u, b_u


def get_t_lower(
    u_c_flat: Tensor,
    l_c_flat: Tensor,
    s_u: Tensor,
    func: TensorFunction = K.sigmoid,
    f_prime: TensorFunction = sigmoid_prime,
) -> tuple[Tensor, Tensor]:
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

    return w_l, b_l
