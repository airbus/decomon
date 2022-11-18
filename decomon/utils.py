from __future__ import absolute_import

import numpy as np
import tensorflow as tf

# from .layers.utils import get_linear_hull_s_shape, sigmoid_prime, tanh_prime
# from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.keras.layers import Flatten, Lambda
from tensorflow.math import greater_equal
from tensorflow.python.keras import backend as K

from .corners.slope import get_linear_lower_slope_relu
from .layers.core import F_FORWARD, F_HYBRID, F_IBP, StaticVariables


# create static variables for varying convex domain
class Ball:
    name = "ball"  # Lp Ball around an example


class Box:
    name = "box"  # Hypercube


class Grid:
    name = "grid"  # Hypercube
    stable_coeff = 0.0


class Vertex:
    name = "vertex"  # convex set represented by its vertices
    # (no verification is proceeded to assess that the set is convex)


# create static variables for varying convex domain
class V_slope:
    name = "volume-slope"


class A_slope:
    name = "adaptative-slope"


class S_slope:
    name = "same-slope"


class Z_slope:
    name = "zero-lb"


class O_slope:
    name = "one-lb"


# propagation of constant and affines bounds from input to output
class M_BACKWARD:
    name = "backward"


class M_FORWARD:
    name = "forward"


class M_REC_BACKWARD:
    name = "crown"


# linear hull for activation function
def relu_prime(x):
    """
    Derivative of relu
    :param x:
    :return:
    """

    return K.clip(K.sign(x), K.cast(0, dtype=x.dtype), K.cast(1, dtype=x.dtype))


def sigmoid_prime(x):
    """
    Derivative of sigmoid
    :param x:
    :return:
    """

    s_x = K.sigmoid(x)
    return s_x * (K.cast(1, dtype=x.dtype) - s_x)


def tanh_prime(x):
    """
    Derivative of tanh
    :param x:
    :return:
    """

    s_x = K.tanh(x)
    return K.cast(1, dtype=x.dtype) - K.pow(s_x, K.cast(2, dtype=x.dtype))


def softsign_prime(x):
    """
    Derivative of softsign
    :param x:
    :return:
    """

    return K.cast(1.0, dtype=x.dtype) / K.pow(K.cast(1.0, dtype=x.dtype) + K.abs(x), K.cast(2, dtype=x.dtype))


##############
# SYMBOLIC UPPER/ LOWER BOUNDS
# compute symbolically constant upper and lower
# with the current knowledge of the convex domain considered
##############

# case 1: a box
def get_upper_box(x_min, x_max, w, b):
    """
    #compute the max of an affine function
    within a box (hypercube) defined by its extremal corners
    :param x_min: lower bound of the box domain
    :param x_max: upper bound of the box domain
    :param w: weights of the affine function
    :param b: bias of the affine function
    :return: max_(x >= x_min, x<=x_max) w*x + b
    """

    if len(w.shape) == len(b.shape):  # identity function
        return x_max

    # split into positive and negative components
    z_value = K.cast(0.0, dtype=x_min.dtype)
    w_pos = K.maximum(w, z_value)
    w_neg = K.minimum(w, z_value)

    x_min_ = x_min + z_value * x_min
    x_max_ = x_max + z_value * x_max

    for _ in range(len(w.shape) - len(x_max.shape)):
        x_min_ = K.expand_dims(x_min_, -1)
        x_max_ = K.expand_dims(x_max_, -1)

    return K.sum(w_pos * x_max_ + w_neg * x_min_, 1) + b


def get_lower_box(x_min, x_max, w, b):
    """

    :param x_min: lower bound of the box domain
    :param x_max: upper bound of the box domain
    :param w_l: weights of the affine lower bound
    :param b_l: bias of the affine lower bound
    :return: min_(x >= x_min, x<=x_max) w*x + b
    """

    if len(w.shape) == len(b.shape):
        return x_min

    z_value = K.cast(0.0, dtype=x_min.dtype)

    w_pos = K.maximum(w, z_value)
    w_neg = K.minimum(w, z_value)

    x_min_ = x_min + z_value * x_min
    x_max_ = x_max + z_value * x_max

    for _ in range(len(w.shape) - len(x_max.shape)):
        x_min_ = K.expand_dims(x_min_, -1)
        x_max_ = K.expand_dims(x_max_, -1)

    return K.sum(w_pos * x_min_ + w_neg * x_max_, 1) + b


# case 2 : a ball
def get_lq_norm(x, p, axis=-1):
    """
    compute Lp norm (p=1 or 2)
    :param x: tensor
    :param p: the power must be an integer in (1, 2)
    :param axis: the axis on which we compute the norm
    :return: ||w||^p
    """
    if p == 1:
        # x_p = K.sum(K.abs(x), axis)
        x_q = K.max(K.abs(x), axis)
    elif p == 2:
        x_q = K.sqrt(K.sum(K.pow(x, p), axis))
    else:
        raise NotImplementedError("p must be equal to 1 or 2")

    return x_q


def get_upper_ball(x_0, eps, p, w, b):
    """
    max of an affine function over an Lp ball
    :param x_0: the center of the ball
    :param eps: the radius
    :param p: the type of Lp norm considered
    :param w: weights of the affine function
    :param b: bias of the affine function
    :return: max_(|x - x_0|_p<= eps) w*x + b
    """
    if len(w.shape) == len(b.shape):
        return x_0 + eps

    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_upper_box(x_min, x_max, w, b)

    else:
        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        upper = eps * get_lq_norm(w, p, axis=1) + b

        for _ in range(len(w.shape) - len(x_0.shape)):
            x_0 = K.expand_dims(x_0, -1)

        return K.sum(w * x_0, 1) + upper


def get_lower_ball(x_0, eps, p, w, b):
    """
    min of an affine fucntion over an Lp ball
    :param x_0: the center of the ball
    :param eps: the radius
    :param p: the type of Lp norm considered
    :param w: weights of the affine function
    :param b: bias of the affine function
    :return: min_(|x - x_0|_p<= eps) w*x + b
    """
    if len(w.shape) == len(b.shape):
        return x_0 - eps

    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_lower_box(x_min, x_max, w, b)

    else:
        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        lower = -eps * get_lq_norm(w, p, axis=1) + b

        for _ in range(len(w.shape) - len(x_0.shape)):
            x_0 = K.expand_dims(x_0, -1)

        return K.sum(w * x_0, 1) + lower


def get_upper(x, w, b, convex_domain=None):
    """
    Meta function that aggregates all the way
    to compute a constant upper bounds depending on the convex domain
    :param x: the tensors that represent the domain
    :param w: the weights of the affine function
    :param b: the bias
    :param convex_domain: the type of convex domain (see ???)
    :return: a constant upper bound of the affine function
    """

    if convex_domain is None:
        convex_domain = {}
    if convex_domain is None or len(convex_domain) == 0:
        # box
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min, x_max, w, b)

    if convex_domain["name"] == Box.name or convex_domain["name"] == Grid.name:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min, x_max, w, b)

    if convex_domain["name"] == Ball.name:

        eps = convex_domain["eps"]
        p = convex_domain["p"]
        return get_upper_ball(x, eps, p, w, b)

    if convex_domain["name"] == Vertex.name:
        raise NotImplementedError()

    raise NotImplementedError()


def get_lower(x, w, b, convex_domain=None):
    """
     Meta function that aggregates all the way
     to compute a constant lower bound depending on the convex domain
    :param x: the tensors that represent the domain
    :param w: the weights of the affine function
    :param b: the bias
    :param convex_domain: the type of convex domain (see ???)
    :return: a constant upper bound of the affine function
    """
    if convex_domain is None:
        convex_domain = {}
    if convex_domain is None or len(convex_domain) == 0:

        # box
        x_min = x[:, 0]
        x_max = x[:, 1]

        return get_lower_box(x_min, x_max, w, b)

    if convex_domain["name"] == Box.name or convex_domain["name"] == Grid.name:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_lower_box(x_min, x_max, w, b)

    if convex_domain["name"] == Ball.name:

        eps = convex_domain["eps"]
        p = convex_domain["p"]

        return get_lower_ball(x, eps, p, w, b)

    if convex_domain["name"] == Vertex.name:
        raise NotImplementedError()

    raise NotImplementedError()


def get_lower_layer(convex_domain=None):
    if convex_domain is None:
        convex_domain = {}

    def func(inputs):
        return get_lower(inputs[0], inputs[1], inputs[2], convex_domain=convex_domain)

    return Lambda(func)


def get_upper_layer(convex_domain=None):
    if convex_domain is None:
        convex_domain = {}

    def func(inputs):
        return get_upper(inputs[0], inputs[1], inputs[2], convex_domain=convex_domain)

    return Lambda(func)


def get_lower_layer_box():
    def func(inputs):
        return get_lower_box(inputs[0], inputs[1], inputs[2], inputs[3])

    return Lambda(func)


def get_upper_layer_box():
    def func(inputs):
        return get_upper_box(inputs[0], inputs[1], inputs[2], inputs[3])

    return Lambda(func)


def backward_maximum(inputs_, convex_domain):

    y, x = inputs_[:2]
    back_bounds_0 = inputs_[2:6]
    back_bounds = inputs_[6:]

    output = inputs_[:2] + back_bounds_0
    for i in range(len(back_bounds) / 4):
        w_u, b_u, w_l, b_l = back_bounds[4 * i : 4 * (i + 1)]
        output = maximum(
            output,
            inputs_[:2] + back_bounds[4 * i : 4 * (i + 1)],
            dc_decomp=False,
            convex_domain=convex_domain,
            mode=F_FORWARD.name,
        )

    return output[-2:]


def backward_minimum(inputs_, convex_domain):

    y, x = inputs_[:2]
    back_bounds_0 = inputs_[2:6]
    back_bounds = inputs_[6:]

    output = inputs_[:2] + back_bounds_0
    for i in range(len(back_bounds) / 4):
        w_u, b_u, w_l, b_l = back_bounds[4 * i : 4 * (i + 1)]
        output = minimum(
            output,
            inputs_[:2] + back_bounds[4 * i : 4 * (i + 1)],
            dc_decomp=False,
            convex_domain=convex_domain,
            mode=F_FORWARD.name,
        )

    return output[2:4]


def noisy_lower(lower):

    # if some random binary variable is set to 0 return K.maximum(upper,- upper)
    var_ = K.minimum(lower, -lower)
    proba = K.random_binomial(lower.shape, p=0.2, dtype=K.floatx())

    return proba * lower + (1 - proba) * var_


def noisy_upper(upper):

    # if some random binary variable is set to 0 return K.maximum(upper,- upper)
    var_ = K.maximum(upper, -upper)
    proba = K.random_binomial(upper.shape, p=0.2, dtype=K.floatx())

    return proba * upper + (1 - proba) * var_


# define routines to get linear relaxations useful both for forward and backward
def get_linear_hull_relu(upper, lower, slope, upper_g=0, lower_g=0, **kwargs):

    # upper = K.in_train_phase(noisy_upper(upper), upper)
    # lower = K.in_train_phase(noisy_lower(upper), lower)

    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    alpha = (K.relu(upper) - K.relu(lower)) / K.maximum(K.cast(K.epsilon(), dtype=upper.dtype), upper - lower)

    # scaling factor for the upper bound on the relu
    # see README

    w_u_ = alpha
    b_u_ = K.relu(lower) - alpha * lower
    z_value = K.cast(0.0, dtype=upper.dtype)
    o_value = K.cast(1.0, dtype=upper.dtype)

    if slope == V_slope.name:
        # 1 if upper<=-lower else 0
        index_a = -K.clip(K.sign(upper + lower) - o_value, -o_value, z_value)

        # 1 if upper>-lower else 0
        index_b = o_value - index_a
        w_l_ = index_b
        b_l_ = z_value * b_u_

    if slope == A_slope.name:
        w_l_ = K.clip(K.sign(w_u_ - 0.5), 0, 1)
        b_l_ = z_value * b_u_

    if slope == Z_slope.name:
        w_l_ = z_value * w_u_
        b_l_ = z_value * b_u_

    if slope == O_slope.name:
        w_l_ = z_value * w_u_ + o_value
        b_l_ = z_value * b_u_

    if slope == S_slope.name:
        w_l_ = w_u_
        b_l_ = z_value * b_u_

    if "upper_grid" in kwargs:

        raise NotImplementedError()
        upper_grid = kwargs["upper_grid"]
        lower_grid = kwargs["lower_grid"]

        w_l_, b_l_ = get_linear_lower_slope_relu(upper, lower, upper_grid, lower_grid, **kwargs)

    if "finetune" in kwargs:
        raise NotImplementedError()
        if not ("finetune_grid" in kwargs and len(kwargs["finetune_grid"])):

            # weighted linear combination
            alpha_l = kwargs["finetune"]
            alpha_l_0 = alpha_l[0][None]
            alpha_l_1 = alpha_l[1][None]

            w_l_ = alpha_l_0 * w_l_ + (1 - alpha_l_0) * alpha_l_1
            b_l_ = alpha_l_0 * b_l_

    # check inactive relu state: u<=0
    index_dead = -K.clip(K.sign(upper) - o_value, -o_value, z_value)  # =1 if inactive state
    index_linear = K.clip(K.sign(lower) + o_value, z_value, o_value)  # 1 if linear state

    w_u_ = (o_value - index_dead) * w_u_
    w_l_ = (o_value - index_dead) * w_l_
    b_u_ = (o_value - index_dead) * b_u_
    b_l_ = (o_value - index_dead) * b_l_

    w_u_ = (o_value - index_linear) * w_u_ + index_linear
    w_l_ = (o_value - index_linear) * w_l_ + index_linear
    b_u_ = (o_value - index_linear) * b_u_
    b_l_ = (o_value - index_linear) * b_l_

    return [w_u_, b_u_, w_l_, b_l_]


def get_linear_hull_sigmoid(upper, lower, slope, **kwargs):

    x = [upper, lower]
    return get_linear_hull_s_shape(
        x, func=K.sigmoid, f_prime=sigmoid_prime, convex_domain={}, mode=F_IBP.name, **kwargs
    )


def get_linear_hull_tanh(upper, lower, slope, **kwargs):

    x = [upper, lower]
    return get_linear_hull_s_shape(x, func=K.tanh, f_prime=tanh_prime, convex_domain={}, mode=F_IBP.name, **kwargs)


def get_linear_softplus_hull(upper, lower, slope, **kwargs):

    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    u_c_ = K.softsign(upper)
    l_c_ = K.softsign(lower)
    alpha = (u_c_ - l_c_) / K.maximum(K.cast(K.epsilon(), dtype=upper.dtype), (upper - lower))
    w_u_ = alpha
    b_u_ = -alpha * lower + l_c_

    z_value = K.cast(0.0, dtype=upper.dtype)
    o_value = K.cast(1.0, dtype=upper.dtype)

    if slope == V_slope.name:
        # 1 if upper<=-lower else 0
        index_a = -K.clip(K.sign(upper + lower) - o_value, -o_value, z_value)
        # 1 if upper>-lower else 0
        index_b = o_value - index_a
        w_l_ = index_b
        b_l_ = z_value * b_u_
    elif slope == Z_slope.name:
        w_l_ = z_value * w_u_
        b_l_ = z_value * b_u_
    elif slope == O_slope.name:
        w_l_ = z_value * w_u_ + o_value
        b_l_ = z_value * b_u_
    elif slope == S_slope.name:
        w_l_ = w_u_
        b_l_ = z_value * b_u_
    else:
        raise ValueError(f"Unknown slope {slope}")

    if "finetune" in kwargs:
        # weighted linear combination
        alpha_l = kwargs["finetune"]
        alpha_l_ = alpha_l[None]

        w_l_ = alpha_l_ * w_l_
        b_l_ = alpha_l_ * b_l_ + (o_value - alpha_l_) * K.maximum(lower, z_value)

    index_dead = -K.clip(K.sign(upper) - o_value, -o_value, z_value)

    w_u_ = (o_value - index_dead) * w_u_
    w_l_ = (o_value - index_dead) * w_l_
    b_u_ = (o_value - index_dead) * b_u_
    b_l_ = (o_value - index_dead) * b_l_

    if "finetune" in kwargs:
        # weighted linear combination
        alpha_u, alpha_l = kwargs["finetune"]
        alpha_u_ = alpha_u[None]
        alpha_l_ = alpha_l[None]

        w_u_ = alpha_u_ * w_u_
        b_u_ = alpha_u_ * b_u_ + (o_value - alpha_u_) * K.maximum(upper, z_value)

        w_l_ = alpha_l_ * w_l_
        b_l_ = alpha_l_ * b_l_ + (o_value - alpha_l_) * K.maximum(lower, z_value)

    return [w_u_, b_u_, w_l_, b_l_]


def maximum(inputs_0, inputs_1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """
    LiRPA implementation of element-wise max

    :param inputs_0: list of tensors
    :param inputs_1: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :return: maximum(inputs_0, inputs_1)
    """
    if convex_domain is None:
        convex_domain = {}
    output_0 = substract(inputs_1, inputs_0, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    output_1 = relu_(
        output_0,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        mode=mode,
    )

    return add(
        output_1,
        inputs_0,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        mode=mode,
    )


def minimum(inputs_0, inputs_1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """
    LiRPA implementation of element-wise min

    :param inputs_0:
    :param inputs_1:
    :param dc_decomp:
    :param convex_domain:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    return minus(maximum(minus(inputs_0), minus(inputs_1), dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode))


def get_linear_hull_s_shape(x, func=K.sigmoid, f_prime=sigmoid_prime, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """
    Computing the linear hull of shape functions  given the pre activation neurons

    :param x: list of input tensors
    :param func: the function (sigmoid, tanh, softsign...)
    :param f_prime: the derivative of the function (sigmoid_prime...)
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: the updated list of tensors
    """

    if convex_domain is None:
        convex_domain = {}
    z_value = K.cast(0.0, dtype=x[0].dtype)
    o_value = K.cast(1.0, dtype=x[0].dtype)
    t_value = K.cast(2.0, dtype=x[0].dtype)

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        # y, x_0, u_c, l_c = x[:4]
        u_c, l_c = x[:nb_tensor]
    elif mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensor]
    elif mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = x[:6]
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensor]
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # flatten
    shape = list(u_c.shape[1:])
    u_c_flat = K.reshape(u_c, (-1, np.prod(shape)))  # (None, n)
    l_c_flat = K.reshape(l_c, (-1, np.prod(shape)))  # (None, n)

    # upper bound
    # derivative
    s_u_prime = f_prime(u_c_flat)  # (None, n)
    s_l_prime = f_prime(l_c_flat)  # (None, n)
    s_u = func(u_c_flat)  # (None, n)
    s_l = func(l_c_flat)  # (None, n)

    # case 0:
    coeff = (s_u - s_l) / K.maximum(K.cast(K.epsilon(), K.floatx()), u_c_flat - l_c_flat)
    alpha_u_0 = K.switch(greater_equal(s_u_prime, coeff), o_value + z_value * u_c_flat, z_value * u_c_flat)  # (None, n)
    alpha_u_1 = (o_value - alpha_u_0) * ((K.sign(l_c_flat) + o_value) / t_value)

    w_u_0 = coeff
    b_u_0 = -w_u_0 * l_c_flat + s_l

    w_u_1 = z_value * u_c_flat
    b_u_1 = s_u

    w_u_2, b_u_2 = get_t_upper(u_c_flat, l_c_flat, s_l, func=func, f_prime=f_prime)

    w_u_ = K.reshape(alpha_u_0 * w_u_0 + alpha_u_1 * w_u_1 + (o_value - alpha_u_0 - alpha_u_1) * w_u_2, [-1] + shape)
    b_u_ = K.reshape(alpha_u_0 * b_u_0 + alpha_u_1 * b_u_1 + (o_value - alpha_u_0 - alpha_u_1) * b_u_2, [-1] + shape)
    # w_u_ = K.reshape(w_u_, [-1]+shape)
    # b_u_ = K.reshape(b_u_, [-1]+shape)

    # linear hull
    # case 0:
    alpha_l_0 = K.switch(greater_equal(s_l_prime, coeff), o_value + z_value * l_c_flat, z_value * l_c_flat)  # (None, n)
    alpha_l_1 = (o_value - alpha_l_0) * ((K.sign(-u_c_flat) + o_value) / t_value)

    w_l_0 = coeff
    b_l_0 = -w_l_0 * u_c_flat + s_u

    w_l_1 = z_value * u_c_flat
    b_l_1 = s_l

    w_l_2, b_l_2 = get_t_lower(u_c_flat, l_c_flat, s_u, func=func, f_prime=f_prime)

    w_l_ = K.reshape(alpha_l_0 * w_l_0 + alpha_l_1 * w_l_1 + (o_value - alpha_l_0 - alpha_l_1) * w_l_2, [-1] + shape)
    b_l_ = K.reshape(alpha_l_0 * b_l_0 + alpha_l_1 * b_l_1 + (o_value - alpha_l_0 - alpha_l_1) * b_l_2, [-1] + shape)
    """
    if "finetune" in kwargs:
        shape_w = [-1] + list(w_u_.shape[1:])
        shape_b = [-1] + list(b_u_.shape[1:])
        op_flat = Flatten()

        w_u_ = op_flat(w_u_)
        w_l_ = op_flat(w_l_)
        b_u_ = op_flat(b_u_)
        b_l_ = op_flat(b_l_)

        # weighted linear combination
        alpha_l = kwargs["finetune"]
        alpha_l_ = alpha_l[None]

        w_l_ = alpha_l_ * w_l_
        b_l_ = alpha_l_ * b_l_ + (o_value - alpha_l_) * op_flat(K.maximum(l_c, z_value))

        w_u_ = K.reshape(w_u_, shape_w)
        w_l_ = K.reshape(w_l_, shape_w)
        b_u_ = K.reshape(b_u_, shape_b)
        b_l_ = K.reshape(b_l_, shape_b)
    """

    return [w_u_, b_u_, w_l_, b_l_]  # what the hell !!!!


def get_t_upper(u_c_flat, l_c_flat, s_l, func=K.sigmoid, f_prime=sigmoid_prime):
    """
    linear interpolation between lower and upper bounds on the function func to have a symbolic approximation of the best
    coefficient for the affine upper bound
    :param u_c_flat: flatten tensor of constant upper bound
    :param l_c_flat: flatten tensor of constant lower bound
    :param s_l: lowest value of the function func on the domain
    :param func: the function (sigmoid, tanh,  softsign)
    :param f_prime: the derivative of the function
    :return: the upper affine bounds in this subcase
    """

    o_value = K.cast(1.0, dtype=u_c_flat.dtype)
    z_value = K.cast(0.0, dtype=u_c_flat.dtype)

    # step1: find t
    u_c_flat_ = K.expand_dims(u_c_flat, -1)  # (None, n , 1)
    l_c_flat_ = K.expand_dims(l_c_flat, -1)  # (None, n,  1)
    t_ = K.cast(np.linspace(0, 1, 100)[None, None, :], K.floatx()) * u_c_flat_  # (None, n , 100)

    s_p_t_ = f_prime(t_)  # (None, n, 100)
    s_t_ = func(t_)  # (None, n, 100)

    score = K.abs(s_p_t_ - (s_t_ - K.expand_dims(s_l, -1)) / (t_ - l_c_flat_))  # (None, n, 100)
    index_ = K.argmin(score, -1)  # (None, n)
    threshold = K.min(score, -1)  # (None, n)

    index_t = K.cast(
        K.switch(K.greater(threshold, z_value * threshold), index_, K.clip(index_ - 1, 0, 100)), dtype=u_c_flat.dtype
    )  # (None, n)
    t_value = K.sum(
        K.switch(
            K.equal(
                o_value * K.cast(np.arange(0, 100)[None, None, :], K.floatx()) + z_value * u_c_flat_,
                K.expand_dims(index_t, -1) + z_value * u_c_flat_,
            ),
            t_,
            z_value * t_,
        ),
        -1,
    )  # (None, n)

    s_t = func(t_value)  # (None, n)
    w_u = (s_t - s_l) / K.maximum(K.cast(K.epsilon(), dtype=u_c_flat.dtype), t_value - l_c_flat)  # (None, n)
    b_u = -w_u * l_c_flat + s_l  # + func(l_c_flat)

    return [w_u, b_u]


def get_t_lower(u_c_flat, l_c_flat, s_u, func=K.sigmoid, f_prime=sigmoid_prime):
    """
    linear interpolation between lower and upper bounds on the function func to have a symbolic approximation of the best
    coefficient for the affine lower bound
    :param u_c_flat: flatten tensor of constant upper bound
    :param l_c_flat: flatten tensor of constant lower bound
    :param s_u: highest value of the function func on the domain
    :param func: the function (sigmoid, tanh,  softsign)
    :param f_prime: the derivative of the function
    :return: the lower affine bounds in this subcase
    """
    z_value = K.cast(0.0, dtype=u_c_flat.dtype)
    o_value = K.cast(1.0, dtype=u_c_flat.dtype)

    # step1: find t
    u_c_flat_ = K.expand_dims(u_c_flat, -1)  # (None, n , 1)
    l_c_flat_ = K.expand_dims(l_c_flat, -1)  # (None, n,  1)
    t_ = K.cast(np.linspace(0, 1.0, 100)[None, None, :], K.floatx()) * l_c_flat_  # (None, n , 100)

    s_p_t_ = f_prime(t_)  # (None, n, 100)
    s_t_ = func(t_)  # (None, n, 100)

    score = K.abs(s_p_t_ - (K.expand_dims(s_u, -1) - s_t_) / (u_c_flat_ - t_))  # (None, n, 100)
    index_ = K.argmin(score, -1)  # (None, n)

    threshold = K.min(score, -1)
    index_t = K.cast(
        K.switch(K.greater(threshold, z_value * threshold), index_, K.clip(index_ + 1, 0, 100)), dtype=u_c_flat.dtype
    )  # (None, n)
    t_value = K.sum(
        K.switch(
            K.equal(
                o_value * K.cast(np.arange(0, 100)[None, None, :], K.floatx()) + z_value * u_c_flat_,
                K.expand_dims(index_t, -1) + z_value * u_c_flat_,
            ),
            t_,
            z_value * t_,
        ),
        -1,
    )

    s_t = func(t_value)  # (None, n)
    w_l = (s_u - s_t) / K.maximum(K.cast(K.epsilon(), dtype=u_c_flat.dtype), u_c_flat - t_value)  # (None, n)
    b_l = -w_l * u_c_flat + s_u  # func(u_c_flat)

    return [w_l, b_l]


def set_mode(x, final_mode, mode, convex_domain=None):
    if convex_domain is None:
        convex_domain = {}

    if final_mode == mode:
        return x

    x_0, u_c, w_u, b_u, l_c, w_l, b_l = None, None, None, None, None, None, None
    if mode == F_IBP.name:
        u_c, l_c = x
        if final_mode != mode:
            raise NotImplementedError(f"If mode if {F_IBP}, final_mode must be also {F_IBP}.")
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x
        if final_mode in [F_IBP.name, F_HYBRID.name]:
            # compute constant bounds
            u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
            l_c = get_lower(x_0, w_u, b_u, convex_domain=convex_domain)
    elif mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x
    else:
        raise ValueError(f"Unknown mode {mode}")

    if final_mode == F_IBP.name:
        return [u_c, l_c]
    elif final_mode == F_FORWARD.name:
        return [x_0, w_u, b_u, w_l, b_l]
    elif final_mode == F_HYBRID.name:
        return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]
    else:
        raise ValueError(f"Unknown final_mode {final_mode}")


def get_AB(model_):
    dico_AB = {}
    convex_domain = model_.convex_domain
    if not (len(convex_domain) and convex_domain["name"] == "grid" and convex_domain["option"] == "milp"):
        return dico_AB

    for layer in model_.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.grid_finetune
    return dico_AB


def get_AB_finetune(model_):
    dico_AB = {}
    convex_domain = model_.convex_domain
    if not (len(convex_domain) and convex_domain["name"] == "grid" and convex_domain["option"] == "milp"):
        return dico_AB

    if not model_.finetune:
        return dico_AB

    for layer in model_.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.alpha_b_l
    return dico_AB
