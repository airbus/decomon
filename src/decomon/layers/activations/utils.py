from collections.abc import Callable
from typing import Any, Union

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
