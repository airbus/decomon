from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten

from decomon.core import (
    BoxDomain,
    ForwardMode,
    GridDomain,
    InputsOutputsSpec,
    PerturbationDomain,
    Slope,
)
from decomon.layers.utils import sort
from decomon.utils import (
    get_linear_hull_relu,
    get_lower,
    get_lower_box,
    get_upper,
    get_upper_box,
    maximum,
    minus,
    relu_,
    subtract,
)


def backward_add(
    inputs_0: List[tf.Tensor],
    inputs_1: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[List[tf.Tensor]]:
    """Backward  LiRPA of inputs_0+inputs_1

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    op_flat = Flatten(dtype=K.floatx())  # pas terrible  a revoir
    nb_tensors = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    if mode == ForwardMode.IBP:
        u_c_0, l_c_0 = inputs_0[:nb_tensors]
        u_c_1, l_c_1 = inputs_1[:nb_tensors]
    elif mode == ForwardMode.HYBRID:
        x, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = inputs_0[:nb_tensors]
        x, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = inputs_1[:nb_tensors]
        u_c_0_tmp = get_upper(x, w_u_0, b_u_0, perturbation_domain=perturbation_domain)
        u_c_1_tmp = get_upper(x, w_u_1, b_u_1, perturbation_domain=perturbation_domain)
        l_c_0_tmp = get_lower(x, w_l_0, b_l_0, perturbation_domain=perturbation_domain)
        l_c_1_tmp = get_lower(x, w_l_1, b_l_1, perturbation_domain=perturbation_domain)
        u_c_0 = K.minimum(u_c_0, u_c_0_tmp)
        u_c_1 = K.minimum(u_c_1, u_c_1_tmp)
        l_c_0 = K.maximum(l_c_0, l_c_0_tmp)
        l_c_1 = K.maximum(l_c_1, l_c_1_tmp)
    elif mode == ForwardMode.AFFINE:
        x, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0[:nb_tensors]
        x, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1[:nb_tensors]
        u_c_0 = get_upper(x, w_u_0, b_u_0, perturbation_domain=perturbation_domain)
        u_c_1 = get_upper(x, w_u_1, b_u_1, perturbation_domain=perturbation_domain)
        l_c_0 = get_lower(x, w_l_0, b_l_0, perturbation_domain=perturbation_domain)
        l_c_1 = get_lower(x, w_l_1, b_l_1, perturbation_domain=perturbation_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    u_c_0 = op_flat(u_c_0)
    u_c_1 = op_flat(u_c_1)
    l_c_0 = op_flat(l_c_0)
    l_c_1 = op_flat(l_c_1)

    upper_0 = get_upper_box(l_c_0, u_c_0, w_u_out, b_u_out)
    upper_1 = get_upper_box(l_c_1, u_c_1, w_u_out, b_u_out)
    lower_0 = get_lower_box(l_c_0, u_c_0, w_l_out, b_l_out)
    lower_1 = get_lower_box(l_c_1, u_c_1, w_l_out, b_l_out)

    w_u_out_0 = w_u_out
    b_u_out_0 = upper_1
    w_l_out_0 = w_l_out
    b_l_out_0 = lower_1

    w_u_out_1 = w_u_out
    b_u_out_1 = upper_0
    w_l_out_1 = w_l_out
    b_l_out_1 = lower_0

    return [[w_u_out_0, b_u_out_0, w_l_out_0, b_l_out_0], [w_u_out_1, b_u_out_1, w_l_out_1, b_l_out_1]]


def merge_with_previous(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
    w_u_out, b_u_out, w_l_out, b_l_out, w_b_u, b_b_u, w_b_l, b_b_l = inputs

    # w_u_out (None, n_h_in, n_h_out)
    # w_b_u (None, n_h_out, n_out)

    # w_u_out_ (None, n_h_in, n_h_out, 1)
    # w_b_u_ (None, 1, n_h_out, n_out)
    # w_u_out_*w_b_u_ (None, n_h_in, n_h_out, n_out)

    # result (None, n_h_in, n_out)

    if len(w_u_out.shape) == 2:
        w_u_out = tf.linalg.diag(w_u_out)

    if len(w_l_out.shape) == 2:
        w_l_out = tf.linalg.diag(w_l_out)

    if len(w_b_u.shape) == 2:
        w_b_u = tf.linalg.diag(w_b_u)

    if len(w_b_l.shape) == 2:
        w_b_l = tf.linalg.diag(w_b_l)

    # import pdb; pdb.set_trace()

    z_value = K.cast(0.0, dtype=w_u_out.dtype)
    w_b_u_pos = K.maximum(w_b_u, z_value)
    w_b_u_neg = K.minimum(w_b_u, z_value)
    w_b_l_pos = K.maximum(w_b_l, z_value)
    w_b_l_neg = K.minimum(w_b_l, z_value)

    w_u = K.batch_dot(w_u_out, w_b_u_pos, (-1, -2)) + K.batch_dot(w_l_out, w_b_u_neg, (-1, -2))
    w_l = K.batch_dot(w_l_out, w_b_l_pos, (-1, -2)) + K.batch_dot(w_u_out, w_b_l_neg, (-1, -2))
    b_u = K.batch_dot(b_u_out, w_b_u_pos, (-1, -2)) + K.batch_dot(b_l_out, w_b_u_neg, (-1, -2)) + b_b_u
    b_l = K.batch_dot(b_l_out, w_b_l_pos, (-1, -2)) + K.batch_dot(b_u_out, w_b_l_neg, (-1, -2)) + b_b_l

    return [w_u, b_u, w_l, b_l]


def backward_relu_(
    inputs: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward  LiRPA of relu

    Args:
        inputs
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        slope
        mode
        fast

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    nb_tensors = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    if mode == ForwardMode.HYBRID:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        upper = u_c
        lower = l_c
    elif mode == ForwardMode.IBP:
        # y, x_0, u_c, l_c = x[:4]
        u_c, l_c = inputs[:nb_tensors]
        upper = u_c
        lower = l_c
    elif mode == ForwardMode.AFFINE:
        # y, x_0, w_u, b_u, w_l, b_l = x[:6]
        x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
        upper = get_upper(x_0, w_u, b_u, perturbation_domain)
        lower = get_lower(x_0, w_l, b_l, perturbation_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    if isinstance(perturbation_domain, GridDomain) and mode != ForwardMode.IBP:

        raise NotImplementedError()

    shape = np.prod(upper.shape[1:])
    upper = K.reshape(upper, [-1, shape])
    lower = K.reshape(lower, [-1, shape])

    z_value = K.cast(0.0, upper.dtype)

    #############
    w_u_tmp, b_u_tmp, w_l_tmp, b_l_tmp = get_linear_hull_relu(upper, lower, slope=slope, **kwargs)

    w_u_tmp = K.expand_dims(w_u_tmp, -1)
    w_l_tmp = K.expand_dims(w_l_tmp, -1)
    b_u_tmp = K.expand_dims(b_u_tmp, -1)
    b_l_tmp = K.expand_dims(b_l_tmp, -1)

    b_u_out = K.sum(K.maximum(w_u_tmp, z_value) * b_u_tmp + K.minimum(w_u_tmp, z_value) * b_l_tmp, 1) + b_u_tmp
    b_l_out = K.sum(K.maximum(w_l_tmp, z_value) * b_l_tmp + K.minimum(w_l_tmp, z_value) * b_u_tmp, 1) + b_l_tmp
    w_u_out = K.maximum(w_u_tmp, z_value) * w_u_tmp + K.minimum(w_u_tmp, z_value) * w_l_tmp
    w_l_out = K.maximum(w_l_tmp, z_value) * w_l_tmp + K.minimum(w_l_tmp, z_value) * w_u_tmp

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def backward_softplus_(
    inputs: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward  LiRPA of relu

    Args:
        inputs
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode
        fast

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    nb_tensors = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    if mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        upper = u_c
        lower = l_c
    elif mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensors]
        upper = u_c
        lower = l_c
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]
        upper = get_upper(x_0, w_u, b_u, perturbation_domain)
        lower = get_lower(x_0, w_l, b_l, perturbation_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    shape = np.prod(upper.shape[1:])
    upper = K.reshape(upper, [-1, shape])
    raise NotImplementedError()


def backward_linear_prod(
    x_0: tf.Tensor,
    bounds_x: List[tf.Tensor],
    back_bounds: List[tf.Tensor],
    perturbation_domain: Optional[PerturbationDomain] = None,
) -> List[tf.Tensor]:
    """Backward  LiRPA of a subroutine prod

    Args:
        bounds_x
        back_bounds

    Returns:

    """

    z_value = K.cast(0.0, x_0.dtype)
    o_value = K.cast(1.0, x_0.dtype)

    w_u_i, b_u_i, w_l_i, b_l_i = bounds_x
    w_u, b_u, w_l, b_l = back_bounds

    if len(w_u_i.shape) > 3:
        n_dim = w_u_i.get_input_shape_at(0)[1]
        w_u_i = K.reshape(w_u_i, (-1, n_dim, n_dim))
        w_l_i = K.reshape(w_l_i, (-1, n_dim, n_dim))
        b_u_i = K.reshape(b_u_i, (-1, n_dim))
        b_l_i = K.reshape(b_l_i, (-1, n_dim))

    x_max = get_upper(x_0, w_u_i - w_l_i, b_u_i - b_l_i, perturbation_domain=perturbation_domain)
    mask_b = o_value - K.sign(x_max)
    mask_a = o_value - mask_b

    w_u_i_expanded = K.expand_dims(K.expand_dims(w_u_i, 1), -1)
    w_l_i_expanded = K.expand_dims(K.expand_dims(w_l_i, 1), -1)
    b_u_i_expanded = K.expand_dims(K.expand_dims(b_u_i, 1), -1)
    b_l_i_expanded = K.expand_dims(K.expand_dims(b_l_i, 1), -1)
    mask_a = K.expand_dims(K.expand_dims(mask_a, 1), -1)
    mask_b = K.expand_dims(K.expand_dims(mask_b, 1), -1)

    w_u_pos = K.maximum(w_u, z_value)
    w_u_neg = K.minimum(w_u, z_value)
    w_l_pos = K.maximum(w_l, z_value)
    w_l_neg = K.minimum(w_l, z_value)

    w_u_pos_expanded = K.expand_dims(w_u_pos, 2)
    w_u_neg_expanded = K.expand_dims(w_u_neg, 2)
    w_l_pos_expanded = K.expand_dims(w_l_pos, 2)
    w_l_neg_expanded = K.expand_dims(w_l_neg, 2)
    mask_a_expanded = K.expand_dims(mask_a, 2)
    mask_b_expanded = K.expand_dims(mask_b, 2)

    w_u_out = K.sum(
        mask_a_expanded * (w_u_pos_expanded * w_u_i_expanded + w_u_neg_expanded * w_l_i_expanded), 3
    ) + K.sum(K.expand_dims(w_u, 2) * mask_b_expanded * w_u_i_expanded, 3)
    w_l_out = K.sum(
        mask_a_expanded * (w_l_pos_expanded * w_l_i_expanded + w_l_neg_expanded * w_u_i_expanded), 3
    ) + K.sum(K.expand_dims(w_l, 2) * mask_b_expanded * w_l_i_expanded, 3)

    b_u_out = (
        K.sum(mask_a * (w_u_pos * b_u_i_expanded + w_u_neg * b_l_i_expanded), 2)
        + K.sum(mask_b * (w_u * b_u_i_expanded), 2)
        + b_u
    )
    b_l_out = (
        K.sum(mask_a * (w_l_pos * b_l_i_expanded + w_l_neg * b_u_i_expanded), 2)
        + K.sum(mask_b * (w_l * b_l_i_expanded), 2)
        + b_l
    )

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def backward_maximum(
    inputs_0: List[tf.Tensor],
    inputs_1: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[List[tf.Tensor]]:
    """Backward  LiRPA of maximum(inputs_0, inputs_1)

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    input_step_a_0 = subtract(inputs_0, inputs_1, dc_decomp=False, perturbation_domain=perturbation_domain, mode=mode)

    input_step_0 = relu_(input_step_a_0, dc_decomp=False, perturbation_domain=perturbation_domain, mode=mode, **kwargs)

    _, bounds_1 = backward_add(
        input_step_0, inputs_1, w_u_out, b_u_out, w_l_out, b_l_out, perturbation_domain=perturbation_domain, mode=mode
    )

    input_step_a_1 = subtract(inputs_1, inputs_0, dc_decomp=False, perturbation_domain=perturbation_domain, mode=mode)
    input_step_1 = relu_(input_step_a_1, dc_decomp=False, perturbation_domain=perturbation_domain, mode=mode, **kwargs)

    _, bounds_0 = backward_add(
        input_step_1, inputs_0, w_u_out, b_u_out, w_l_out, b_l_out, perturbation_domain=perturbation_domain, mode=mode
    )

    return [bounds_0, bounds_1]


# convex hull of the maximum between two functions
def backward_max_(
    inputs: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Backward  LiRPA of max

    Args:
        inputs: list of tensors
        dc_decomp: boolean that indicates
        grad_bounds: boolean that indicates whether
        perturbation_domain: the type of perturbation domain
        axis: axis to perform the maximum
    whether we return a difference of convex decomposition of our layer
    we propagate upper and lower bounds on the values of the gradient

    Returns:
        max operation  along an axis
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    nb_tensor = InputsOutputsSpec(dc_decomp=False, mode=mode).nb_tensors
    z_value = K.cast(0.0, inputs[0].dtype)

    if mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensor]
        y = u_c
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
        y = b_u
    elif mode == ForwardMode.IBP:
        u_c, l_c = inputs[:nb_tensor]
        y = u_c
    else:
        raise ValueError(f"Unknown mode {mode}")

    input_shape = K.int_shape(y)
    max_dim = input_shape[axis]

    # do some transpose so that the last axis is also at the end
    if mode in [ForwardMode.HYBRID, ForwardMode.IBP]:

        u_c_list = tf.split(u_c, max_dim, axis)
        l_c_list = tf.split(l_c, max_dim, axis)
        u_c_tmp = u_c_list[0] + z_value * (u_c_list[0])
        l_c_tmp = l_c_list[0] + z_value * (l_c_list[0])

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:

        b_u_list = tf.split(b_u, max_dim, axis)
        b_l_list = tf.split(b_l, max_dim, axis)
        b_u_tmp = b_u_list[0] + z_value * (b_u_list[0])
        b_l_tmp = b_l_list[0] + z_value * (b_l_list[0])

        if axis == -1:
            w_u_list = tf.split(w_u, max_dim, axis)
            w_l_list = tf.split(w_l, max_dim, axis)
        else:
            w_u_list = tf.split(w_u, max_dim, axis + 1)
            w_l_list = tf.split(w_l, max_dim, axis + 1)
        w_u_tmp = w_u_list[0] + z_value * (w_u_list[0])
        w_l_tmp = w_l_list[0] + z_value * (w_l_list[0])

    outputs = []
    output_tmp = []  # store output at every level
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
            outputs.append([[elem for elem in output_tmp], output_i])
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    if mode == ForwardMode.IBP:
        output_tmp = [
            u_c_tmp,
            l_c_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [u_c_list[i], l_c_list[i]]
            outputs.append([[elem for elem in output_tmp], output_i])
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    if mode == ForwardMode.AFFINE:
        output_tmp = [
            x_0,
            w_u_tmp,
            b_u_tmp,
            w_l_tmp,
            b_l_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
            outputs.append([[elem for elem in output_tmp], output_i])
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    outputs = outputs[::-1]
    bounds = []

    if len(outputs) > 0:
        for (input_0, input_1) in outputs:
            bounds_0, bounds_1 = backward_maximum(
                input_0, input_1, w_u_out, b_u_out, w_l_out, b_l_out, mode=mode, **kwargs
            )
            bounds.append(bounds_1)
            w_u_out, b_u_out, w_l_out, b_l_out = bounds_0
        bounds.append(bounds_0)
        bounds = bounds[::-1]

    if axis < 0:
        w_u_out = K.concatenate([b[0] for b in bounds], axis - 1)
        w_l_out = K.concatenate([b[2] for b in bounds], axis - 1)
        b_u_out = K.sum(K.concatenate([K.expand_dims(b[1], axis - 1) for b in bounds], axis - 1), axis - 1)
        b_l_out = K.sum(K.concatenate([K.expand_dims(b[3], axis - 1) for b in bounds], axis - 1), axis - 1)
    else:
        w_u_out = K.concatenate([b[0] for b in bounds], axis)
        w_l_out = K.concatenate([b[2] for b in bounds], axis)
        b_u_out = K.sum(K.concatenate([K.expand_dims(b[1], axis) for b in bounds], axis), axis)
        b_l_out = K.sum(K.concatenate([K.expand_dims(b[3], axis) for b in bounds], axis), axis)

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def backward_minimum(
    inputs_0: List[tf.Tensor],
    inputs_1: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    **kwargs: Any,
) -> List[List[tf.Tensor]]:
    """Backward  LiRPA of minimum(inputs_0, inputs_1)

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    w_u_out, b_u_out, w_l_out, b_l_out = backward_minus(w_u_out, b_u_out, w_l_out, b_l_out)
    bounds_0, bounds_1 = backward_maximum(
        inputs_0,
        inputs_1,
        w_u_out,
        b_u_out,
        w_l_out,
        b_l_out,
        perturbation_domain=perturbation_domain,
        mode=mode,
        **kwargs,
    )

    w_u_out_0, b_u_out_0, w_l_out_0, b_l_out_0 = bounds_0
    w_u_out_1, b_u_out_1, w_l_out_1, b_l_out_1 = bounds_1

    bounds_0 = backward_minus(w_u_out_0, b_u_out_0, w_l_out_0, b_l_out_0)
    bounds_1 = backward_minus(w_u_out_1, b_u_out_1, w_l_out_1, b_l_out_1)

    return [bounds_0, bounds_1]


def backward_minus(
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
) -> List[tf.Tensor]:
    """Backward  LiRPA of -x

    Args:
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """
    return [-w_l_out, -b_l_out, -w_u_out, -b_u_out]


def backward_scale(
    scale_factor: float,
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
) -> List[tf.Tensor]:
    """Backward  LiRPA of scale_factor*x

    Args:
        scale_factor
        w_u_out
        b_u_out
        w_l_out
        b_l_out

    Returns:

    """

    if scale_factor >= 0:
        output = [scale_factor * w_u_out, b_u_out, scale_factor * w_l_out, b_l_out]
    else:
        output = [scale_factor * w_l_out, b_l_out, scale_factor * w_u_out, b_u_out]

    return output


def backward_subtract(
    inputs_0: List[tf.Tensor],
    inputs_1: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[List[tf.Tensor]]:
    """Backward  LiRPA of inputs_0 - inputs_1

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    inputs_1 = minus(inputs_1, mode=mode)
    bounds_0, bounds_1 = backward_add(
        inputs_0, inputs_1, w_u_out, b_u_out, w_l_out, b_l_out, perturbation_domain=perturbation_domain, mode=mode
    )

    bounds_1 = [-bounds_1[0], bounds_1[1], -bounds_1[2], bounds_1[3]]
    return [bounds_0, bounds_1]


def backward_multiply(
    inputs_0: List[tf.Tensor],
    inputs_1: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[List[tf.Tensor]]:
    """Backward  LiRPA of element-wise multiply inputs_0*inputs_1

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    if mode == ForwardMode.IBP:
        u_0, l_0 = inputs_0
        u_1, l_1 = inputs_1
    elif mode == ForwardMode.HYBRID:
        x_0, u_0, w_u_0, b_u_0, l_0, w_l_0, b_l_0 = inputs_0
        x_1, u_1, w_u_1, b_u_1, l_1, w_l_1, b_l_1 = inputs_1
    elif mode == ForwardMode.AFFINE:
        x_0, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0
        x_1, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1
        u_0 = get_upper(x_0, w_u_0, b_u_0, perturbation_domain=perturbation_domain)
        l_0 = get_lower(x_0, w_l_0, b_l_0, perturbation_domain=perturbation_domain)
        u_1 = get_upper(x_1, w_u_1, b_u_1, perturbation_domain=perturbation_domain)
        l_1 = get_lower(x_1, w_l_1, b_l_1, perturbation_domain=perturbation_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    z_value = K.cast(0.0, u_0.dtype)

    n = np.prod(u_0.shape[1:])
    n_shape = [-1, n]
    # broadcast dimensions if needed
    n_out = len(w_u_out.shape[1:])
    for _ in range(n_out):
        n_shape += [1]
    a_u_0 = K.reshape(u_1, n_shape)
    a_u_1 = K.reshape(u_0, n_shape)

    b_u_0 = K.reshape((K.maximum(l_0, z_value) * u_1 + K.minimum(l_0, z_value) * l_1 - u_1 * l_0), n_shape)
    b_u_1 = K.reshape((K.maximum(l_1, z_value) * u_0 + K.minimum(l_1, z_value) * l_0 - u_0 * l_1), n_shape)

    a_l_0 = K.reshape(l_1, n_shape)
    a_l_1 = K.reshape(l_0, n_shape)
    b_l_0 = K.reshape((K.maximum(l_0, z_value) * l_1 + K.minimum(l_0, z_value) * u_1 - l_1 * l_0), n_shape)
    b_l_1 = K.reshape((K.maximum(l_1, z_value) * l_0 + K.minimum(l_1, z_value) * u_0 - l_0 * l_1), n_shape)

    # upper
    w_u_out_max = K.maximum(w_u_out, z_value)
    w_u_out_min = K.minimum(w_u_out, z_value)

    w_u_out_0 = w_u_out_max * a_u_0 + w_u_out_min * a_l_0
    w_u_out_1 = w_u_out_max * a_u_1 + w_u_out_min * a_l_1

    b_u_out_0 = K.sum(w_u_out_max * b_u_0, 1) + K.sum(w_u_out_min * b_l_0, 1) + b_u_out
    b_u_out_1 = K.sum(w_u_out_max * b_u_1, 1) + K.sum(w_u_out_min * b_l_1, 1) + b_u_out

    # lower
    w_l_out_max = K.maximum(w_l_out, z_value)
    w_l_out_min = K.minimum(w_l_out, z_value)

    w_l_out_0 = w_l_out_max * a_l_0 + w_l_out_min * a_u_0
    w_l_out_1 = w_l_out_max * a_l_1 + w_l_out_min * a_u_1

    b_l_out_0 = K.sum(w_l_out_max * b_l_0, 1) + K.sum(w_l_out_min * b_u_0, 1) + b_l_out
    b_l_out_1 = K.sum(w_l_out_max * b_l_1, 1) + K.sum(w_l_out_min * b_u_1, 1) + b_l_out

    bounds_0 = [w_u_out_0, b_u_out_0, w_l_out_0, b_l_out_0]
    bounds_1 = [w_u_out_1, b_u_out_1, w_l_out_1, b_l_out_1]

    return [bounds_0, bounds_1]


def backward_sort(
    inputs: List[tf.Tensor],
    w_u_out: tf.Tensor,
    b_u_out: tf.Tensor,
    w_l_out: tf.Tensor,
    b_l_out: tf.Tensor,
    axis: int = -1,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
) -> List[tf.Tensor]:
    """Backward  LiRPA of sort

    Args:
        inputs
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        axis
        perturbation_domain
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    z_value = K.cast(0.0, w_u_out.dtype)

    # build the tightest contain bounds for inputs_
    if mode == ForwardMode.IBP:
        u_c, l_c = inputs
        y = u_c
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs
        y = b_u
        u_c_0 = get_upper(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
        l_c_0 = get_lower(x_0, w_l, b_l, perturbation_domain=perturbation_domain)
        u_c = u_c_0
        l_c = l_c_0
    elif mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs
        y = u_c
        u_c_0 = get_upper(x_0, w_u, b_u, perturbation_domain=perturbation_domain)
        l_c_0 = get_lower(x_0, w_l, b_l, perturbation_domain=perturbation_domain)
        u_c = K.minimum(u_c, u_c_0)
        l_c = K.maximum(l_c, l_c_0)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # build fake inputs with no linearity
    n_dim = np.prod(y.shape[1:])
    w_tmp = z_value * K.concatenate([y[:, None] * n_dim], 1)

    inputs_tmp = [x_0, u_c, w_tmp, u_c, l_c, w_tmp, l_c]
    outputs_tmp = sort(inputs_tmp, axis=axis, perturbation_domain=perturbation_domain, mode=ForwardMode.HYBRID)
    _, _, w_u_tmp, b_u_tmp, _, w_l_tmp, b_l_tmp = outputs_tmp

    # w_u_tmp (None, n_dim, y.shape[1:)
    # w_u_out (None, 1, n_dim, n_out)
    w_u_tmp = K.reshape(w_u_tmp, [-1, n_dim, n_dim])  # (None, n_dim, n_dim)
    b_u_tmp = K.reshape(b_u_tmp, [-1, n_dim])  # (None, n_dim)
    w_l_tmp = K.reshape(w_l_tmp, [-1, n_dim, n_dim])
    b_l_tmp = K.reshape(b_l_tmp, [-1, n_dim])

    # combine with backward bounds
    w_u_out_pos = K.maximum(w_u_out, z_value)  # (None, 1, n_dim, n_out)
    w_u_out_neg = K.minimum(w_u_out, z_value)
    w_l_out_pos = K.maximum(w_l_out, z_value)
    w_l_out_neg = K.minimum(w_l_out, z_value)

    w_u_out = K.sum(w_u_out_pos * K.expand_dims(w_u_tmp, -1) + w_u_out_pos * K.expand_dims(w_l_tmp, -1), 1)
    w_l_out = K.sum(w_u_out_pos * K.expand_dims(w_u_tmp, -1) + w_u_out_pos * K.expand_dims(w_l_tmp, -1), 1)
    b_u_out = b_u_out + K.sum(w_u_out_pos * b_u_tmp, 1) + K.sum(w_u_out_neg * b_l_tmp, 1)
    b_l_out = b_l_out + K.sum(w_l_out_pos * b_l_tmp, 1) + K.sum(w_l_out_neg * b_u_tmp, 1)

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def get_identity_lirpa(inputs: List[tf.Tensor]) -> List[tf.Tensor]:
    y = inputs[-1]
    shape = np.prod(y.shape[1:])

    z_value = K.cast(0.0, y.dtype)
    o_value = K.cast(1.0, y.dtype)
    y_flat = K.reshape(y, [-1, shape])

    w_u_out, w_l_out = [o_value + z_value * y_flat] * 2
    b_u_out, b_l_out = [z_value * y_flat] * 2
    w_u_out = tf.linalg.diag(w_u_out)
    w_l_out = tf.linalg.diag(w_l_out)

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def get_ibp(mode: Union[str, ForwardMode] = ForwardMode.HYBRID) -> bool:
    mode = ForwardMode(mode)
    if mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
        return True
    return False


def get_affine(mode: Union[str, ForwardMode] = ForwardMode.HYBRID) -> bool:
    mode = ForwardMode(mode)
    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        return True
    return False
