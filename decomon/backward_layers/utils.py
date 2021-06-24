from decomon.layers import get_lower, get_upper
from tensorflow.python.keras import backend as K
from ..layers import F_HYBRID, F_IBP, F_FORWARD
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from ..layers.utils import get_upper, get_lower, add, maximum, get_upper_box, get_lower_box, relu_, minus


# create static variables for varying convex domain
class V_slope:
    name = "volume-slope"


class S_slope:
    name = "same-slope"


class Z_slope:
    name = "zero-lb"


class O_slope:
    name = "one-lb"


def backward_minus(inputs_0, inputs_1, w_out_u_, b_out_u_, w_out_l_, b_out_l_, convex_domain={}, mode=F_HYBRID.name):

    op_flat = Flatten()

    if mode == F_IBP.name:
        _, _, u_c_0, l_c_0 = inputs_0
        _, _, u_c_1, l_c_1 = inputs_1

    if mode == F_HYBRID.name:

        _, _, u_c_0, _, _, l_c_0 = inputs_0[:6]
        _, _, u_c_1, _, _, l_c_1 = inputs_1[:6]

    if mode == F_FORWARD.name:

        _, x, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0[:6]
        _, x, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1[:6]

        u_c_0 = get_upper(x, w_u_0, b_u_0, convex_domain=convex_domain)
        u_c_1 = get_upper(x, w_u_1, b_u_1, convex_domain=convex_domain)
        l_c_0 = get_lower(x, w_l_0, b_l_0, convex_domain=convex_domain)
        l_c_1 = get_lower(x, w_l_1, b_l_1, convex_domain=convex_domain)

    u_c_0 = op_flat(u_c_0)
    u_c_1 = op_flat(u_c_1)
    l_c_0 = op_flat(l_c_0)
    l_c_1 = op_flat(l_c_1)
    # we need to flatten !

    upper_0 = get_upper_box(l_c_0, u_c_0, w_out_u_[:, 0], b_out_l_[:, 0])
    upper_1 = get_upper_box(l_c_1, u_c_1, -w_out_u_[:, 0], b_out_u_[:, 0])
    lower_0 = get_lower_box(l_c_0, u_c_0, w_out_u_[:, 0], b_out_u_[:, 0])
    lower_1 = get_lower_box(l_c_1, u_c_1, -w_out_l_[:, 0], b_out_l_[:, 0])

    w_out_u_0 = w_out_u_
    w_out_l_0 = w_out_l_
    b_out_u_0 = upper_1[:, None]
    b_out_l_0 = lower_1[:, None]

    w_out_u_1 = -w_out_l_
    w_out_l_1 = -w_out_u_
    b_out_u_1 = upper_0[:, None]
    b_out_l_1 = lower_0[:, None]

    return [w_out_u_0, b_out_u_0, w_out_l_0, b_out_l_0], [w_out_u_1, b_out_u_1, w_out_l_1, b_out_l_1]


def backward_add(inputs_0, inputs_1, w_out_u_, b_out_u_, w_out_l_, b_out_l_, convex_domain={}, mode=F_HYBRID.name):

    op_flat = Flatten()

    if mode == F_IBP.name:
        _, _, u_c_0, l_c_0 = inputs_0
        _, _, u_c_1, l_c_1 = inputs_1

    if mode == F_HYBRID.name:

        _, _, u_c_0, _, _, l_c_0 = inputs_0[:6]
        _, _, u_c_1, _, _, l_c_1 = inputs_1[:6]

    if mode == F_FORWARD.name:

        _, x, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0[:6]
        _, x, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1[:6]

        u_c_0 = get_upper(x, w_u_0, b_u_0, convex_domain=convex_domain)
        u_c_1 = get_upper(x, w_u_1, b_u_1, convex_domain=convex_domain)
        l_c_0 = get_lower(x, w_l_0, b_l_0, convex_domain=convex_domain)
        l_c_1 = get_lower(x, w_l_1, b_l_1, convex_domain=convex_domain)

    u_c_0 = op_flat(u_c_0)
    u_c_1 = op_flat(u_c_1)
    l_c_0 = op_flat(l_c_0)
    l_c_1 = op_flat(l_c_1)
    # we need to flatten !

    # import pdb; pdb.set_trace()
    upper_0 = get_upper_box(l_c_0, u_c_0, w_out_u_[:, 0], b_out_u_[:, 0])
    upper_1 = get_upper_box(l_c_1, u_c_1, w_out_u_[:, 0], b_out_u_[:, 0])
    lower_0 = get_lower_box(l_c_0, u_c_0, w_out_l_[:, 0], b_out_l_[:, 0])
    lower_1 = get_lower_box(l_c_1, u_c_1, w_out_l_[:, 0], b_out_l_[:, 0])

    w_out_u_0 = w_out_u_
    b_out_u_0 = upper_1[:, None]
    w_out_l_0 = w_out_l_
    b_out_l_0 = lower_1[:, None]

    w_out_u_1 = w_out_u_
    b_out_u_1 = upper_0[:, None]
    w_out_l_1 = w_out_l_
    b_out_l_1 = lower_0[:, None]

    return [w_out_u_0, b_out_u_0, w_out_l_0, b_out_l_0], [w_out_u_1, b_out_u_1, w_out_l_1, b_out_l_1]


def backward_relu_(
    x, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, fast=True
):

    op_flat = Flatten()

    # get input bounds
    if mode == F_HYBRID.name:
        _, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x

        # compute upper and lower bounds
        # and keep the minimum between the constant and the computed upper bounds
        # and keep the maximum between the constant and the computer lower bounds
        if not fast:
            upper = K.minimum(u_c, get_upper(x_0, w_u, b_u, convex_domain))
            lower = K.maximum(l_c, get_lower(x_0, w_l, b_l, convex_domain))
        else:
            upper = u_c
            lower = l_c
    elif mode == F_IBP.name:
        _, x_0, u_c, l_c = x
        upper = u_c
        lower = l_c
    elif mode == F_FORWARD.name:
        _, x_0, w_u, b_u, w_l, b_l = x
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)

    # check inactive relu state: u<=0
    index_dead = -K.clip(K.sign(upper), -1, 0)
    index_linear = K.clip(K.sign(lower) + 1, 0, 1)

    # 1 if upper<=-lower else 0
    index_a = -K.clip(K.sign(upper + lower) - 1, -1.0, 0.0)
    # 1 if upper>-lower else 0
    index_b = K.ones_like(index_a) - index_a

    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    w_u_0 = upper / K.maximum(K.epsilon(), upper - lower)
    b_u_0 = -upper * lower / K.maximum(K.epsilon(), upper - lower)

    w_u_ = (
        index_dead * K.zeros_like(w_u_0) + index_linear * K.ones_like(w_u_0) + (1.0 - index_dead - index_linear) * w_u_0
    )
    b_u_ = (index_dead + index_linear) * K.zeros_like(b_u_0) + (1 - index_linear - index_dead) * b_u_0

    # lower bound
    b_l_ = K.zeros_like(b_u_)
    if slope == V_slope.name:
        w_l_ = K.zeros_like(w_u_)
        w_l_ += K.maximum(index_b, index_linear) * K.ones_like(w_l_)

    if slope == Z_slope.name:
        w_l_ = K.zeros_like(w_u_)

    if slope == O_slope.name:
        w_l_ = K.ones_like(w_u_)

    if slope == S_slope.name:
        w_l_ = w_u_

    w_u_ = op_flat(w_u_)
    w_l_ = op_flat(w_l_)
    b_u_ = op_flat(b_u_)
    b_l_ = op_flat(b_l_)

    w_u_ = K.expand_dims(K.expand_dims(w_u_, 1), -1)
    w_l_ = K.expand_dims(K.expand_dims(w_l_, 1), -1)
    b_u_ = K.expand_dims(K.expand_dims(b_u_, 1), -1)
    b_l_ = K.expand_dims(K.expand_dims(b_l_, 1), -1)

    w_out_u_ = K.maximum(w_out_u, 0.0) * w_u_ + K.minimum(w_out_u, 0.0) * w_l_
    w_out_l_ = K.maximum(w_out_l, 0.0) * w_l_ + K.minimum(w_out_l, 0.0) * w_u_
    b_out_u_ = K.sum(K.maximum(w_out_u, 0.0) * b_u_ + K.minimum(w_out_u, 0.0) * b_l_, 2) + b_out_u
    b_out_l_ = K.sum(K.maximum(w_out_l, 0.0) * b_l_ + K.minimum(w_out_l, 0.0) * b_u_, 2) + b_out_l

    return w_out_u_, b_out_u_, w_out_l_, b_out_l_


def backward_linear_prod(x_0, bounds_x, back_bounds, convex_domain):
    """

    :param bounds_x:
    :param back_bounds:
    :return:
    """
    w_u_i, b_u_i, w_l_i, b_l_i = bounds_x
    w_u, b_u, w_l, b_l = back_bounds

    if len(w_u_i.shape) > 3:
        n_dim = w_u_i.get_input_shape_at(0)[1]
        w_u_i = K.reshape(w_u_i, (-1, n_dim, n_dim))
        w_l_i = K.reshape(w_l_i, (-1, n_dim, n_dim))
        b_u_i = K.reshape(b_u_i, (-1, n_dim))
        b_l_i = K.reshape(b_l_i, (-1, n_dim))

    # reshape bounds_x !!!!

    x_max = get_upper(x_0, w_u_i - w_l_i, b_u_i - b_l_i, convex_domain)
    mask_b = 1.0 - K.sign(x_max)
    mask_a = 1.0 - mask_b

    w_u_i_ = K.expand_dims(K.expand_dims(w_u_i, 1), -1)
    w_l_i_ = K.expand_dims(K.expand_dims(w_l_i, 1), -1)
    b_u_i_ = K.expand_dims(K.expand_dims(b_u_i, 1), -1)
    b_l_i_ = K.expand_dims(K.expand_dims(b_l_i, 1), -1)
    mask_a = K.expand_dims(K.expand_dims(mask_a, 1), -1)
    mask_b = K.expand_dims(K.expand_dims(mask_b, 1), -1)

    w_u_pos = K.maximum(w_u, 0.0)
    w_u_neg = K.minimum(w_u, 0.0)
    w_l_pos = K.maximum(w_l, 0.0)
    w_l_neg = K.minimum(w_l, 0.0)

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
    inputs_0, inputs_1, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name
):
    input_tmp_a = minus(inputs_1, inputs_0, dc_decomp=False, grad_bounds=False, convex_domain=convex_domain, mode=mode)
    input_tmp = relu_(
        input_tmp_a,
        dc_decomp=False,
        grad_bounds=False,
        convex_domain=convex_domain,
        mode=mode,
    )

    # start with add

    bounds_add_0, bounds_add_tmp = backward_add(
        inputs_0, input_tmp, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=convex_domain, mode=mode
    )

    bounds_add_tmp = backward_relu_(
        input_tmp_a,
        bounds_add_tmp[0],
        bounds_add_tmp[1],
        bounds_add_tmp[2],
        bounds_add_tmp[3],
        slope=slope,
        convex_domain=convex_domain,
        mode=mode,
    )

    bounds_add_1, bounds_add_0_1 = backward_minus(
        inputs_1,
        inputs_0,
        bounds_add_tmp[0],
        bounds_add_tmp[1],
        bounds_add_tmp[2],
        bounds_add_tmp[3],
        convex_domain=convex_domain,
        mode=mode,
    )

    return bounds_add_0, bounds_add_1


# convex hull of the maximum between two functions
def backward_max_(
    x, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, axis=-1
):
    """

    :param x: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates whether
    we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :param axis: axis to perform the maximum
    :return: max operation  along an axis
    """

    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
    if mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = x[:6]
    if mode == F_IBP.name:
        y, x_0, u_c, l_c = x[:4]

    input_shape = K.int_shape(y)
    max_dim = input_shape[axis]

    # do some transpose so that the last axis is also at the end

    y_list = tf.split(y, max_dim, axis)
    y_tmp = y_list[0] + K.zeros_like(y_list[0])

    if mode in [F_HYBRID.name, F_IBP.name]:

        u_c_list = tf.split(u_c, max_dim, axis)
        l_c_list = tf.split(l_c, max_dim, axis)
        u_c_tmp = u_c_list[0] + K.zeros_like(u_c_list[0])
        l_c_tmp = l_c_list[0] + K.zeros_like(l_c_list[0])

    if mode in [F_HYBRID.name, F_FORWARD.name]:

        b_u_list = tf.split(b_u, max_dim, axis)
        b_l_list = tf.split(b_l, max_dim, axis)
        b_u_tmp = b_u_list[0] + K.zeros_like(b_u_list[0])
        b_l_tmp = b_l_list[0] + K.zeros_like(b_l_list[0])

        if axis == -1:
            w_u_list = tf.split(w_u, max_dim, axis)
            w_l_list = tf.split(w_l, max_dim, axis)
        else:
            w_u_list = tf.split(w_u, max_dim, axis + 1)
            w_l_list = tf.split(w_l, max_dim, axis + 1)
        w_u_tmp = w_u_list[0] + K.zeros_like(w_u_list[0])
        w_l_tmp = w_l_list[0] + K.zeros_like(w_l_list[0])

    outputs = []
    output_tmp = []  # store output at every level
    if mode == F_HYBRID.name:
        output_tmp = [
            y_tmp,
            x_0,
            u_c_tmp,
            w_u_tmp,
            b_u_tmp,
            l_c_tmp,
            w_l_tmp,
            b_l_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [y_list[i], x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]
            outputs.append([[elem for elem in output_tmp], output_i])
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    if mode == F_IBP.name:
        output_tmp = [
            y_tmp,
            x_0,
            u_c_tmp,
            l_c_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [y_list[i], x_0, u_c_list[i], l_c_list[i]]
            outputs.append([[elem for elem in output_tmp], output_i])
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    if mode == F_FORWARD.name:
        output_tmp = [
            y_tmp,
            x_0,
            w_u_tmp,
            b_u_tmp,
            w_l_tmp,
            b_l_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [y_list[i], x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
            outputs.append([[elem for elem in output_tmp], output_i])
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    outputs = outputs[::-1]
    bounds = []
    w_u_, b_u_, w_l_, b_l_ = (w_out_u, b_out_u, w_out_l, b_out_l)

    for (input_0, input_1) in outputs:
        bounds_0, bounds_1 = backward_maximum(input_0, input_1, w_u_, b_u_, w_l_, b_l_, mode=mode)
        bounds.append(bounds_1)
        w_u_, b_u_, w_l_, b_l_ = bounds_0
    bounds.append(bounds_0)
    bounds = bounds[::-1]

    w_u_ = K.concatenate([K.expand_dims(b[0], -1) for b in bounds], -1)
    w_l_ = K.concatenate([K.expand_dims(b[2], -1) for b in bounds], -1)
    b_u_ = K.sum(K.concatenate([K.expand_dims(b[1], -1) for b in bounds], -1), -1)
    b_l_ = K.sum(K.concatenate([K.expand_dims(b[3], -1) for b in bounds], -1), -1)

    return [w_u_, b_u_, w_l_, b_l_]
