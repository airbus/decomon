import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras import backend as K

from decomon.layers import F_FORWARD, F_HYBRID, F_IBP, StaticVariables
from decomon.layers.core import Grid
from decomon.layers.utils import sort
from decomon.utils import (
    V_slope,
    get_linear_hull_relu,
    get_lower,
    get_lower_box,
    get_upper,
    get_upper_box,
    maximum,
    minus,
    relu_,
    substract,
)


def backward_add(inputs_0, inputs_1, w_out_u_, b_out_u_, w_out_l_, b_out_l_, convex_domain=None, mode=F_HYBRID.name):
    """
    Backward  LiRPA of inputs_0+inputs_1
    :param inputs_0:
    :param inputs_1:
    :param w_out_u_:
    :param b_out_u_:
    :param w_out_l_:
    :param b_out_l_:
    :param convex_domain:
    :param mode:
    :return:
    """
    if convex_domain is None:
        convex_domain = {}
    op_flat = Flatten(dtype=K.floatx())  # pas terrible  a revoir
    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        u_c_0, l_c_0 = inputs_0[:nb_tensors]
        u_c_1, l_c_1 = inputs_1[:nb_tensors]
    elif mode == F_HYBRID.name:
        x, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = inputs_0[:nb_tensors]
        x, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = inputs_1[:nb_tensors]
        u_c_0_ = get_upper(x, w_u_0, b_u_0, convex_domain=convex_domain)
        u_c_1_ = get_upper(x, w_u_1, b_u_1, convex_domain=convex_domain)
        l_c_0_ = get_lower(x, w_l_0, b_l_0, convex_domain=convex_domain)
        l_c_1_ = get_lower(x, w_l_1, b_l_1, convex_domain=convex_domain)
        u_c_0 = K.minimum(u_c_0, u_c_0_)
        u_c_1 = K.minimum(u_c_1, u_c_1_)
        l_c_0 = K.maximum(l_c_0, l_c_0_)
        l_c_1 = K.maximum(l_c_1, l_c_1_)
    elif mode == F_FORWARD.name:
        x, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0[:nb_tensors]
        x, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1[:nb_tensors]
        u_c_0 = get_upper(x, w_u_0, b_u_0, convex_domain=convex_domain)
        u_c_1 = get_upper(x, w_u_1, b_u_1, convex_domain=convex_domain)
        l_c_0 = get_lower(x, w_l_0, b_l_0, convex_domain=convex_domain)
        l_c_1 = get_lower(x, w_l_1, b_l_1, convex_domain=convex_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    u_c_0 = op_flat(u_c_0)
    u_c_1 = op_flat(u_c_1)
    l_c_0 = op_flat(l_c_0)
    l_c_1 = op_flat(l_c_1)

    upper_0 = get_upper_box(l_c_0, u_c_0, w_out_u_, b_out_u_)
    upper_1 = get_upper_box(l_c_1, u_c_1, w_out_u_, b_out_u_)
    lower_0 = get_lower_box(l_c_0, u_c_0, w_out_l_, b_out_l_)
    lower_1 = get_lower_box(l_c_1, u_c_1, w_out_l_, b_out_l_)

    w_out_u_0 = w_out_u_
    b_out_u_0 = upper_1
    w_out_l_0 = w_out_l_
    b_out_l_0 = lower_1

    w_out_u_1 = w_out_u_
    b_out_u_1 = upper_0
    w_out_l_1 = w_out_l_
    b_out_l_1 = lower_0

    return [w_out_u_0, b_out_u_0, w_out_l_0, b_out_l_0], [w_out_u_1, b_out_u_1, w_out_l_1, b_out_l_1]


def merge_with_previous(inputs):
    w_out_u, b_out_u, w_out_l, b_out_l, w_b_u, b_b_u, w_b_l, b_b_l = inputs

    # w_out_u (None, n_h_in, n_h_out)
    # w_b_u (None, n_h_out, n_out)

    # w_out_u_ (None, n_h_in, n_h_out, 1)
    # w_b_u_ (None, 1, n_h_out, n_out)
    # w_out_u_*w_b_u_ (None, n_h_in, n_h_out, n_out)

    # result (None, n_h_in, n_out)

    if len(w_out_u.shape) == 2:
        w_out_u = tf.linalg.diag(w_out_u)

    if len(w_out_l.shape) == 2:
        w_out_l = tf.linalg.diag(w_out_l)

    if len(w_b_u.shape) == 2:
        w_b_u = tf.linalg.diag(w_b_u)

    if len(w_b_l.shape) == 2:
        w_b_l = tf.linalg.diag(w_b_l)

    w_b_u_ = K.expand_dims(w_b_u, 1)
    w_b_l_ = K.expand_dims(w_b_l, 1)
    w_out_u_ = K.expand_dims(w_out_u, -1)
    w_out_l_ = K.expand_dims(w_out_l, -1)
    b_out_u_ = K.expand_dims(b_out_u, -1)
    b_out_l_ = K.expand_dims(b_out_l, -1)

    z_value = K.cast(0.0, dtype=w_out_u.dtype)

    w_u = K.sum(K.maximum(w_b_u_, z_value) * w_out_u_ + K.minimum(w_b_u_, z_value) * w_out_l_, 2)
    w_l = K.sum(K.maximum(w_b_l_, z_value) * w_out_l_ + K.minimum(w_b_l_, z_value) * w_out_u_, 2)
    b_u = K.sum(K.maximum(w_b_u, z_value) * b_out_u_ + K.minimum(w_b_u, z_value) * b_out_l_, 1) + b_b_u
    b_l = K.sum(K.maximum(w_b_l, z_value) * b_out_l_ + K.minimum(w_b_l, z_value) * b_out_u_, 1) + b_b_l

    return [w_u, b_u, w_l, b_l]


def backward_relu_(
    x,
    w_out_u,
    b_out_u,
    w_out_l,
    b_out_l,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_HYBRID.name,
    fast=True,
    **kwargs,
):
    """
    Backward  LiRPA of relu
    :param x:
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :param convex_domain:
    :param slope:
    :param mode:
    :param fast:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensors]
        upper = u_c
        lower = l_c
    elif mode == F_IBP.name:
        # y, x_0, u_c, l_c = x[:4]
        u_c, l_c = x[:nb_tensors]
        upper = u_c
        lower = l_c
    elif mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = x[:6]
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensors]
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    if len(convex_domain) and convex_domain["name"] == Grid.name and mode != F_IBP.name:

        raise NotImplementedError()

    shape = np.prod(upper.shape[1:])
    upper = K.reshape(upper, [-1, shape])
    lower = K.reshape(lower, [-1, shape])

    z_value = K.cast(0.0, upper.dtype)

    #############
    w_u_, b_u_, w_l_, b_l_ = get_linear_hull_relu(upper, lower, slope=slope, **kwargs)

    w_u_ = K.expand_dims(w_u_, -1)
    w_l_ = K.expand_dims(w_l_, -1)
    b_u_ = K.expand_dims(b_u_, -1)
    b_l_ = K.expand_dims(b_l_, -1)

    w_out_u_ = K.maximum(w_out_u, z_value) * w_u_ + K.minimum(w_out_u, z_value) * w_l_
    w_out_l_ = K.maximum(w_out_l, z_value) * w_l_ + K.minimum(w_out_l, z_value) * w_u_
    b_out_u_ = K.sum(K.maximum(w_out_u, z_value) * b_u_ + K.minimum(w_out_u, z_value) * b_l_, 1) + b_out_u
    b_out_l_ = K.sum(K.maximum(w_out_l, z_value) * b_l_ + K.minimum(w_out_l, z_value) * b_u_, 1) + b_out_l

    return w_out_u_, b_out_u_, w_out_l_, b_out_l_


def backward_softplus_(
    x,
    w_out_u,
    b_out_u,
    w_out_l,
    b_out_l,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_HYBRID.name,
    fast=True,
    **kwargs,
):
    """
    Backward  LiRPA of relu
    :param x:
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :param convex_domain:
    :param slope:
    :param mode:
    :param fast:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensors]
        upper = u_c
        lower = l_c
    elif mode == F_IBP.name:
        u_c, l_c = x[:nb_tensors]
        upper = u_c
        lower = l_c
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensors]
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    shape = np.prod(upper.shape[1:])
    upper = K.reshape(upper, [-1, shape])


def backward_linear_prod(x_0, bounds_x, back_bounds, convex_domain):
    """
    Backward  LiRPA of a subroutine prod
    :param bounds_x:
    :param back_bounds:
    :return:
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

    x_max = get_upper(x_0, w_u_i - w_l_i, b_u_i - b_l_i, convex_domain)
    mask_b = o_value - K.sign(x_max)
    mask_a = o_value - mask_b

    w_u_i_ = K.expand_dims(K.expand_dims(w_u_i, 1), -1)
    w_l_i_ = K.expand_dims(K.expand_dims(w_l_i, 1), -1)
    b_u_i_ = K.expand_dims(K.expand_dims(b_u_i, 1), -1)
    b_l_i_ = K.expand_dims(K.expand_dims(b_l_i, 1), -1)
    mask_a = K.expand_dims(K.expand_dims(mask_a, 1), -1)
    mask_b = K.expand_dims(K.expand_dims(mask_b, 1), -1)

    w_u_pos = K.maximum(w_u, z_value)
    w_u_neg = K.minimum(w_u, z_value)
    w_l_pos = K.maximum(w_l, z_value)
    w_l_neg = K.minimum(w_l, z_value)

    w_u_pos_ = K.expand_dims(w_u_pos, 2)
    w_u_neg_ = K.expand_dims(w_u_neg, 2)
    w_l_pos_ = K.expand_dims(w_l_pos, 2)
    w_l_neg_ = K.expand_dims(w_l_neg, 2)
    mask_a_ = K.expand_dims(mask_a, 2)
    mask_b_ = K.expand_dims(mask_b, 2)

    w_u_ = K.sum(mask_a_ * (w_u_pos_ * w_u_i_ + w_u_neg_ * w_l_i_), 3) + K.sum(
        K.expand_dims(w_u, 2) * mask_b_ * w_u_i_, 3
    )
    w_l_ = K.sum(mask_a_ * (w_l_pos_ * w_l_i_ + w_l_neg_ * w_u_i_), 3) + K.sum(
        K.expand_dims(w_l, 2) * mask_b_ * w_l_i_, 3
    )

    b_u_ = K.sum(mask_a * (w_u_pos * b_u_i_ + w_u_neg * b_l_i_), 2) + K.sum(mask_b * (w_u * b_u_i_), 2) + b_u
    b_l_ = K.sum(mask_a * (w_l_pos * b_l_i_ + w_l_neg * b_u_i_), 2) + K.sum(mask_b * (w_l * b_l_i_), 2) + b_l

    return [w_u_, b_u_, w_l_, b_l_]


def backward_maximum(
    inputs_0,
    inputs_1,
    w_out_u,
    b_out_u,
    w_out_l,
    b_out_l,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_HYBRID.name,
    **kwargs,
):
    """
    Backward  LiRPA of maximum(inputs_0, inputs_1)
    :param inputs_0:
    :param inputs_1:
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    input_step_a_0 = substract(inputs_0, inputs_1, dc_decomp=False, convex_domain=convex_domain, mode=mode)

    input_step_0_ = relu_(input_step_a_0, dc_decomp=False, convex_domain=convex_domain, mode=mode, **kwargs)

    _, bounds_1_ = backward_add(
        input_step_0_, inputs_1, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=convex_domain, mode=mode
    )

    input_step_a_1 = substract(inputs_1, inputs_0, dc_decomp=False, convex_domain=convex_domain, mode=mode)
    input_step_1_ = relu_(input_step_a_1, dc_decomp=False, convex_domain=convex_domain, mode=mode, **kwargs)

    _, bounds_0_ = backward_add(
        input_step_1_, inputs_0, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=convex_domain, mode=mode
    )

    return bounds_0_, bounds_1_


# convex hull of the maximum between two functions
def backward_max_(
    x, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=None, slope=V_slope.name, mode=F_HYBRID.name, axis=-1, **kwargs
):
    """
    Backward  LiRPA of max
    :param x: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates whether
    we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :param axis: axis to perform the maximum
    :return: max operation  along an axis
    """
    if convex_domain is None:
        convex_domain = {}
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    z_value = K.cast(0.0, x.dtype)
    x_0, b_u, b_l, w_u, w_l, u_c, l_c = None, None, None, None, None, None, None
    u_c_tmp, w_u_tmp, b_u_tmp, l_c_tmp, w_l_tmp, b_l_tmp = None, None, None, None, None, None
    u_c_list, w_u_list, b_u_list, l_c_list, w_l_list, b_l_list = None, None, None, None, None, None
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensor]
        y = u_c
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensor]
        y = b_u
    elif mode == F_IBP.name:
        u_c, l_c = x[:nb_tensor]
        y = u_c
    else:
        raise ValueError(f"Unknown mode {mode}")

    input_shape = K.int_shape(y)
    max_dim = input_shape[axis]

    # do some transpose so that the last axis is also at the end
    if mode in [F_HYBRID.name, F_IBP.name]:

        u_c_list = tf.split(u_c, max_dim, axis)
        l_c_list = tf.split(l_c, max_dim, axis)
        u_c_tmp = u_c_list[0] + z_value * (u_c_list[0])
        l_c_tmp = l_c_list[0] + z_value * (l_c_list[0])

    if mode in [F_HYBRID.name, F_FORWARD.name]:

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
    if mode == F_HYBRID.name:
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

    if mode == F_IBP.name:
        output_tmp = [
            u_c_tmp,
            l_c_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [u_c_list[i], l_c_list[i]]
            outputs.append([[elem for elem in output_tmp], output_i])
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    if mode == F_FORWARD.name:
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
    w_u_, b_u_, w_l_, b_l_ = (w_out_u, b_out_u, w_out_l, b_out_l)

    if len(outputs) > 0:
        bounds_0 = None
        for (input_0, input_1) in outputs:
            bounds_0, bounds_1 = backward_maximum(input_0, input_1, w_u_, b_u_, w_l_, b_l_, mode=mode, **kwargs)
            bounds.append(bounds_1)
            w_u_, b_u_, w_l_, b_l_ = bounds_0
        bounds.append(bounds_0)
        bounds = bounds[::-1]

    if axis < 0:
        w_u_ = K.concatenate([b[0] for b in bounds], axis - 1)
        w_l_ = K.concatenate([b[2] for b in bounds], axis - 1)
        b_u_ = K.sum(K.concatenate([K.expand_dims(b[1], axis - 1) for b in bounds], axis - 1), axis - 1)
        b_l_ = K.sum(K.concatenate([K.expand_dims(b[3], axis - 1) for b in bounds], axis - 1), axis - 1)
    else:
        w_u_ = K.concatenate([b[0] for b in bounds], axis)
        w_l_ = K.concatenate([b[2] for b in bounds], axis)
        b_u_ = K.sum(K.concatenate([K.expand_dims(b[1], axis) for b in bounds], axis), axis)
        b_l_ = K.sum(K.concatenate([K.expand_dims(b[3], axis) for b in bounds], axis), axis)

    return [w_u_, b_u_, w_l_, b_l_]


def backward_minimum(
    inputs_0,
    inputs_1,
    w_out_u,
    b_out_u,
    w_out_l,
    b_out_l,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_HYBRID.name,
    **kwargs,
):
    """
    Backward  LiRPA of minimum(inputs_0, inputs_1)
    :param inputs_0:
    :param inputs_1:
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = backward_minus(
        w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=convex_domain, slope=slope, mode=mode
    )
    bounds_0, bounds_1 = backward_maximum(
        inputs_0,
        inputs_1,
        w_out_u_,
        b_out_u_,
        w_out_l_,
        b_out_l_,
        convex_domain=convex_domain,
        slope=slope,
        mode=mode,
        **kwargs,
    )

    w_out_u_0, b_out_u_0, w_out_l_0, b_out_l_0 = bounds_0
    w_out_u_1, b_out_u_1, w_out_l_1, b_out_l_1 = bounds_1

    bounds_0_ = backward_minus(
        w_out_u_0, b_out_u_0, w_out_l_0, b_out_l_0, convex_domain=convex_domain, slope=slope, mode=mode
    )
    bounds_1_ = backward_minus(
        w_out_u_1, b_out_u_1, w_out_l_1, b_out_l_1, convex_domain=convex_domain, slope=slope, mode=mode
    )

    return bounds_0_, bounds_1_


def backward_minus(w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=None, slope=V_slope.name, mode=F_HYBRID.name):
    """
    Backward  LiRPA of -x
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    w_u_ = -w_out_l
    b_u_ = -b_out_l
    w_l_ = -w_out_u
    b_l_ = -b_out_u
    return [w_u_, b_u_, w_l_, b_l_]


def backward_scale(scale_factor, w_out_u, b_out_u, w_out_l, b_out_l):
    """
    Backward  LiRPA of scale_factor*x
    :param scale_factor:
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :return:
    """

    if scale_factor >= 0:
        output = [scale_factor * w_out_u, b_out_u, scale_factor * w_out_l, b_out_l]
    else:
        output = [scale_factor * w_out_l, b_out_u, scale_factor * w_out_l, b_out_l]

    return output


def backward_substract(
    inputs_0, inputs_1, w_out_u_, b_out_u_, w_out_l_, b_out_l_, convex_domain=None, mode=F_HYBRID.name
):
    """
    Backward  LiRPA of inputs_0 - inputs_1
    :param inputs_0:
    :param inputs_1:
    :param w_out_u_:
    :param b_out_u_:
    :param w_out_l_:
    :param b_out_l_:
    :param convex_domain:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    inputs_1_ = minus(inputs_1, mode=mode)
    bounds_0, bounds_1 = backward_add(
        inputs_0, inputs_1_, w_out_u_, b_out_u_, w_out_l_, b_out_l_, convex_domain=convex_domain, mode=mode
    )

    bounds_1_ = [-bounds_1[0], bounds_1[1], -bounds_1[2], bounds_1[3]]
    return bounds_0, bounds_1_


def backward_multiply(
    inputs_0, inputs_1, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=None, slope=V_slope.name, mode=F_HYBRID.name
):
    """
    Backward  LiRPA of element-wise multiply inputs_0*inputs_1
    :param inputs_0:
    :param inputs_1:
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    if mode == F_IBP.name:
        u_0, l_0 = inputs_0
        u_1, l_1 = inputs_1
    elif mode == F_HYBRID.name:
        x_0, u_0, w_u_0, b_u_0, l_0, w_l_0, b_l_0 = inputs_0
        x_1, u_1, w_u_1, b_u_1, l_1, w_l_1, b_l_1 = inputs_1
    elif mode == F_FORWARD.name:
        x_0, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0
        x_1, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1
        u_0 = get_upper(x_0, w_u_0, b_u_0, convex_domain=convex_domain)
        l_0 = get_lower(x_0, w_l_0, b_l_0, convex_domain=convex_domain)
        u_1 = get_upper(x_1, w_u_1, b_u_1, convex_domain=convex_domain)
        l_1 = get_lower(x_1, w_l_1, b_l_1, convex_domain=convex_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    z_value = K.cast(0.0, u_0.dtype)

    n = np.prod(u_0.shape[1:])
    n_shape = [-1, n]
    # broadcast dimensions if needed
    n_out = len(w_out_u.shape[1:])
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
    w_out_u_max = K.maximum(w_out_u, z_value)
    w_out_u_min = K.minimum(w_out_u, z_value)

    w_out_u_0 = w_out_u_max * a_u_0 + w_out_u_min * a_l_0
    w_out_u_1 = w_out_u_max * a_u_1 + w_out_u_min * a_l_1

    b_out_u_0 = K.sum(w_out_u_max * b_u_0, 1) + K.sum(w_out_u_min * b_l_0, 1) + b_out_u
    b_out_u_1 = K.sum(w_out_u_max * b_u_1, 1) + K.sum(w_out_u_min * b_l_1, 1) + b_out_u

    # lower
    w_out_l_max = K.maximum(w_out_l, z_value)
    w_out_l_min = K.minimum(w_out_l, z_value)

    w_out_l_0 = w_out_l_max * a_l_0 + w_out_l_min * a_u_0
    w_out_l_1 = w_out_l_max * a_l_1 + w_out_l_min * a_u_1

    b_out_l_0 = K.sum(w_out_l_max * b_l_0, 1) + K.sum(w_out_l_min * b_u_0, 1) + b_out_l
    b_out_l_1 = K.sum(w_out_l_max * b_l_1, 1) + K.sum(w_out_l_min * b_u_1, 1) + b_out_l

    bounds_0 = [w_out_u_0, b_out_u_0, w_out_l_0, b_out_l_0]
    bounds_1 = [w_out_u_1, b_out_u_1, w_out_l_1, b_out_l_1]

    return bounds_0, bounds_1


def backward_sort(
    inputs_,
    w_out_u,
    b_out_u,
    w_out_l,
    b_out_l,
    axis=-1,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_HYBRID.name,
    **kwargs,
):
    """
    Backward  LiRPA of sort
    :param inputs_:
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :param axis:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """
    if convex_domain is None:
        convex_domain = {}
    z_value = K.cast(0.0, w_out_u.dtype)

    # build the tightest contain bounds for inputs_
    if mode == F_IBP.name:
        u_c_, l_c_ = inputs_
        y = u_c_
    elif mode == F_FORWARD.name:
        x_0, w_u_, b_u_, w_l_, b_l_ = inputs_
        y = b_u_
        u_c_0 = get_upper(x_0, w_u_, b_u_, convex_domain=convex_domain)
        l_c_0 = get_lower(x_0, w_l_, b_l_, convex_domain=convex_domain)
        u_c_ = u_c_0
        l_c_ = l_c_0
    elif mode == F_HYBRID.name:
        x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = inputs_
        y = u_c_
        u_c_0 = get_upper(x_0, w_u_, b_u_, convex_domain=convex_domain)
        l_c_0 = get_lower(x_0, w_l_, b_l_, convex_domain=convex_domain)
        u_c_ = K.minimum(u_c_, u_c_0)
        l_c_ = K.maximum(l_c_, l_c_0)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # build fake inputs with no linearity
    n_dim = np.prod(y.shape[1:])
    w_tmp = z_value * K.concatenate([y[:, None] * n_dim], 1)

    inputs_tmp = [x_0, u_c_, w_tmp, u_c_, l_c_, w_tmp, l_c_]
    outputs_tmp = sort(inputs_tmp, axis=axis, convex_domain=convex_domain, mode=F_HYBRID.name)
    _, _, w_u_tmp, b_u_tmp, _, w_l_tmp, b_l_tmp = outputs_tmp

    # w_u_tmp (None, n_dim, y.shape[1:)
    # w_out_u (None, 1, n_dim, n_out)
    w_u_tmp = K.reshape(w_u_tmp, [-1, n_dim, n_dim])  # (None, n_dim, n_dim)
    b_u_tmp = K.reshape(b_u_tmp, [-1, n_dim])  # (None, n_dim)
    w_l_tmp = K.reshape(w_l_tmp, [-1, n_dim, n_dim])
    b_l_tmp = K.reshape(b_l_tmp, [-1, n_dim])

    # combine with backward bounds
    w_out_u_pos = K.maximum(w_out_u, z_value)  # (None, 1, n_dim, n_out)
    w_out_u_neg = K.minimum(w_out_u, z_value)
    w_out_l_pos = K.maximum(w_out_l, z_value)
    w_out_l_neg = K.minimum(w_out_l, z_value)

    w_out_u_ = K.sum(w_out_u_pos * K.expand_dims(w_u_tmp, -1) + w_out_u_pos * K.expand_dims(w_l_tmp, -1), 1)
    w_out_l_ = K.sum(w_out_u_pos * K.expand_dims(w_u_tmp, -1) + w_out_u_pos * K.expand_dims(w_l_tmp, -1), 1)
    b_out_u_ = b_out_u + K.sum(w_out_u_pos * b_u_tmp, 1) + K.sum(w_out_u_neg * b_l_tmp, 1)
    b_out_l_ = b_out_l + K.sum(w_out_l_pos * b_l_tmp, 1) + K.sum(w_out_l_neg * b_u_tmp, 1)

    return [w_out_u_, b_out_u_, w_out_l_, b_out_l_]


def get_identity_lirpa(inputs):
    y_ = inputs[-1]
    shape = np.prod(y_.shape[1:])

    z_value = K.cast(0.0, y_.dtype)
    o_value = K.cast(1.0, y_.dtype)
    y_flat = K.reshape(y_, [-1, shape])

    w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
    b_out_u, b_out_l = [z_value * y_flat] * 2
    w_out_u = tf.linalg.diag(w_out_u)
    w_out_l = tf.linalg.diag(w_out_l)

    return w_out_u, b_out_u, w_out_l, b_out_l


def get_IBP(mode=F_HYBRID.name):
    if mode in [F_HYBRID.name, F_IBP.name]:
        return True
    return False


def get_FORWARD(mode=F_HYBRID.name):
    if mode in [F_HYBRID.name, F_FORWARD.name]:
        return True
    return False


def get_input_dim(input_dim, convex_domain):
    if len(convex_domain) == 0:
        return 2, input_dim
    else:
        return input_dim