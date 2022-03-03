from __future__ import absolute_import
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
import tensorflow.keras as keras
import numpy as np
from .core import Ball, Box, Vertex, F_FORWARD, F_IBP, F_HYBRID, StaticVariables
from tensorflow.math import greater_equal
from tensorflow.keras.layers import Flatten

# from ..utils import get_linear_hull_relu, get_linear_softplus_hull
from ..utils import *


# create static variables for varying convex domain
class V_slope:
    name = "volume-slope"


class S_slope:
    name = "same-slope"


class Z_slope:
    name = "zero-lb"


class O_slope:
    name = "one-lb"


"""
def _get_shape_tuple(init_tuple, tensor, start_idx, int_shape=None):
    ""Finds non-specific dimensions in the static shapes.
    The static shapes are replaced with the corresponding dynamic shapes of the
    tensor.
    Arguments:
      init_tuple: a tuple, the first part of the output shape
      tensor: the tensor from which to get the (static and dynamic) shapes
        as the last part of the output shape
      start_idx: int, which indicate the first dimension to take from
        the static shape of the tensor
      int_shape: an alternative static shape to take as the last part
        of the output shape
    Returns:
      The new int_shape with the first part from init_tuple
      and the last part from either `int_shape` (if provided)
      or `tensor.shape`, where every `None` is replaced by
      the corresponding dimension from `tf.shape(tensor)`.

    sources: official Keras directory
    ""
    # replace all None in int_shape by K.shape
    if int_shape is None:
        int_shape = K.int_shape(tensor)[start_idx:]
    if not any(not s for s in int_shape):
        return init_tuple + tuple(int_shape)
    shape = K.shape(tensor)
    int_shape = list(int_shape)
    for i, s in enumerate(int_shape):
        if not s:
            int_shape[i] = shape[start_idx + i]
    return init_tuple + tuple(int_shape)
"""

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

    z_value = K.cast(0.0, K.floatx())

    if len(w.shape) == len(b.shape):  # identity function
        return x_max

    # split into positive and negative components
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
    z_value = K.cast(0.0, K.floatx())

    if len(w.shape) == len(b.shape):
        return x_min

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

    if p not in [1, 2]:
        raise NotImplementedError()

    if p == 1:
        # x_p = K.sum(K.abs(x), axis)
        x_q = K.max(K.abs(x), axis)
    if p == 2:
        x_q = K.sqrt(K.sum(K.pow(x, p), axis))

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


def get_upper(x, w, b, convex_domain={}):
    """
    Meta function that aggregates all the way
    to compute a constant upper bounds depending on the convex domain
    :param x: the tensors that represent the domain
    :param w: the weights of the affine function
    :param b: the bias
    :param convex_domain: the type of convex domain (see ???)
    :return: a constant upper bound of the affine function
    """
    if convex_domain is None or len(convex_domain) == 0:
        # box
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min, x_max, w, b)

    if convex_domain["name"] == Box.name:
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


def get_lower(x, w, b, convex_domain={}):
    """
     Meta function that aggregates all the way
     to compute a constant lower bound depending on the convex domain
    :param x: the tensors that represent the domain
    :param w: the weights of the affine function
    :param b: the bias
    :param convex_domain: the type of convex domain (see ???)
    :return: a constant upper bound of the affine function
    """
    if convex_domain is None or len(convex_domain) == 0:

        # box
        x_min = x[:, 0]
        x_max = x[:, 1]

        return get_lower_box(x_min, x_max, w, b)

    if convex_domain["name"] == Box.name:
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


#####
# USE SYMBOLIC GRADIENT DESCENT WITH OVERESTIMATION GUARANTEES
#####
# first compute gradient of the function
def get_grad(x, constant, W, b):
    """
    We compute the gradient of the function f at sample x

    f = sum_{i<= n_linear} max(constant_i, W_i*x + b_i)
    it is quite easy to compute the gradient symbolically, without using the gradient operator
    it is either 0 if f(x) = constant or W else
    this trick allows to be backpropagation compatible

    :param x: Keras Tensor (None, n_dim, ...)
    :param constant: Keras Tensor (None, n_linear, ...)
    :param W: Keras Tensor (None, n_dim, n_linear, ...)
    :param b: Keras Tensor (None, n_linear, ...)
    :return: Keras Tensor the gradient of the function (None, n_dim, ...)
    """
    # product W*x + b
    # import pdb; pdb.set_trace()
    # x_ = K.expand_dims(x, 2)  # (None, n_dim, 1, 1)
    # z = K.sum(W * x_, 1) + b

    x_ = K.expand_dims(x, 2)
    z = K.sum(W * x_, 1) + b

    # grad_ = K.sum(K.expand_dims(K.clip(K.sign(K.maximum(constant, z) - constant), 0., 1.), 1)*W, 2)

    grad_ = K.sum(K.expand_dims(-K.sign(constant - K.maximum(constant, z)), 1) * W, 2)  # (None, n_dim, ...)

    return grad_


def compute_L(W):
    """
    We compute the largest possible norm of the gradient
    :param W: Keras Tensor (None, n_dim, n_linear, ...)
    :return: Keras Tensor with an upper bound on the largest magnitude of the gradient
    """
    # do L2 norm
    return K.sum(K.sqrt(K.sum(W * W, 1)), 1)

    # return K.sum(K.sqrt(K.sum(K.pow(W, 2), 1)), 1)


def compute_R(z, convex_domain):
    """
    We compute the largest L2 distance of the starting point with the global optimum
    :param z: Keras Tensor
    :param convex_domain: Dictionnary to complement z on the convex domain
    :return: Keras Tensor an upper bound on the distance
    """

    if len(convex_domain) == 0:
        # compute the L2 distance z[:, 0], z[:, 1]
        dist_ = K.sqrt(K.sum(K.pow((z[:, 1] - z[:, 0]) / 2.0, 2), -1))
    elif convex_domain["name"] == Box.name:  # to improve
        dist_ = K.sqrt(K.sum(K.pow((z[:, 1] - z[:, 0]) / 2.0, 2), -1))
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == np.inf:
        dist_ = K.sqrt(K.sum(K.pow(z - z + convex_domain["eps"], 2), -1))
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == 2:
        dist_ = convex_domain["eps"] * (0 * z + 1.0)
    else:
        raise NotImplementedError()

    return dist_


def get_start_point(z, convex_domain):
    """
    Create a warm start for the optimization (the mid point to minimize the largest distance between the
    warm start and the global optimum
    :param z: Keras Tensor
    :param convex_domain: Dictionnary to complement z on the convex domain
    :return: Keras Tensor
    """

    if len(convex_domain) == 0:
        # compute the L2 distance z[:, 0], z[:, 1]
        # return (z[:, 0] + z[:, 1]) / 2.0
        return z[:, 0]
    elif convex_domain["name"] == Box.name:  # to improve
        return (z[:, 0] + z[:, 1]) / 2.0
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == np.inf:
        return z
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == 2:
        return z
    else:
        raise NotImplementedError()


def get_coeff_grad(R, k, g):
    """

    :param R: Keras Tensor that reprends the largest distance to the gloabl optimum
    :param k: the number of iteration done so far
    :param g: the gradient
    :return: the adaptative step size for the gradient
    """

    denum = np.sqrt(k) * K.sqrt(K.sum(K.pow(g, 2), 1))

    alpha = R / K.maximum(K.epsilon(), denum)
    return alpha


def grad_descent_conv(z, concave_upper, convex_lower, op_pos, ops_neg, n_iter):
    """

    :param z:
    :param concave_upper:
    :param convex_lower:
    :param op_pos:
    :param ops_neg:
    :param n_iter:
    :return:
    """

    raise NotImplementedError()


def grad_descent(z, convex_0, convex_1, convex_domain, n_iter=5):
    """

    :param z: Keras Tensor
    :param constant: Keras Tensor, the constant of each component
    :param W: Keras Tensor, the affine of each component
    :param b: Keras Tensor, the bias of each component
    :param convex_domain: Dictionnary to complement z on the convex domain
    :param n_iter: the number of total iteration
    :return:
    """

    constant_0, W_0, b_0 = convex_0
    constant_1, W_1, b_1 = convex_1

    # init
    # import pdb; pdb.set_trace()
    x_k = K.expand_dims(get_start_point(z, convex_domain), -1) + 0 * K.sum(constant_0, 1)[:, None]
    R = compute_R(z, convex_domain)
    n_dim = len(x_k.shape[1:])
    while n_dim > 1:
        R = K.expand_dims(R, -1)
        n_dim -= 1

    def step_grad(x_, x_k_):

        x_k = x_k_[0]
        g_k_0 = get_grad(x_k, constant_0, W_0, b_0)
        g_k_1 = get_grad(x_k, constant_1, W_1, b_1)
        g_k = g_k_0 + g_k_1
        alpha_k = get_coeff_grad(R, n_iter + 1, g_k)
        # x_result = x_k - alpha_k* g_k
        x_result = alpha_k[:, None] * g_k
        return x_result, [x_k]

    # step_grad(x_k, [x_k])
    x_vec = K.rnn(
        step_function=step_grad, inputs=K.concatenate([x_k[:, None]] * n_iter, 1), initial_states=[x_k], unroll=False
    )[1]

    # check convergence
    x_k = x_vec[:, -1]
    g_k = get_grad(x_k, constant_0, W_0, b_0) + get_grad(x_k, constant_1, W_1, b_1)
    mask_grad = K.sign(K.sqrt(K.sum(K.pow(g_k, 2), 1)))  # check whether we have converge
    X_vec = K.expand_dims(x_vec, -2)
    f_vec = K.sum(
        K.maximum(constant_0[:, None], K.sum(W_0[:, None] * X_vec, 2) + b_0[:, None])
        + K.maximum(constant_1[:, None], K.sum(W_1[:, None] * X_vec, 2) + b_1[:, None]),
        2,
    )
    # f_vec = K.min(f_vec, 1)
    f_vec = f_vec[0]
    L_0 = compute_L(W_0)
    L_1 = compute_L(W_1)
    L = L_0 + L_1
    penalty = (L * R) / np.sqrt(n_iter + 1)
    return f_vec - mask_grad * penalty


class NonPos(Constraint):
    """Constrains the weights to be non-negative."""

    def __call__(self, w):
        return w * K.cast(K.less_equal(w, 0.0), K.floatx())


class ClipAlpha(Constraint):
    """Cosntraints the weights to be between 0 and 1."""

    def __call__(self, w):
        return K.clip(w, 0.0, 1.0)

class ClipAlphaAndSumtoOne(Constraint):
    """Cosntraints the weights to be between 0 and 1."""

    def __call__(self, w):
        w = K.clip(w, 0.0, 1.0)
        # normalize the first colum to 1
        w_scale = K.maximum(K.sum(w, 0), K.epsilon())
        return w/w_scale[:,None,None,None]


class MultipleConstraint(Constraint):
    """
    stacking multiple constraints
    """

    def __init__(self, constraint_0, constraint_1, **kwargs):
        super(MultipleConstraint, self).__init__(**kwargs)
        if constraint_0:
            self.constraints = [constraint_0, constraint_1]
        else:
            self.constraints = [constraint_1]

    def __call__(self, w):
        w_ = w
        for c in self.constraints:
            w_ = c.__call__(w_)

        return w_


class Project_initializer_pos(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, initializer, **kwargs):
        super(Project_initializer_pos, **kwargs)
        self.initializer = initializer

    def __call__(self, shape, dtype=None):
        w_ = self.initializer.__call__(shape, dtype)
        return K.maximum(0.0, w_)


class Project_initializer_neg(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, initializer, **kwargs):
        super(Project_initializer_neg, **kwargs)
        self.initializer = initializer

    def __call__(self, shape, dtype=None):
        w_ = self.initializer.__call__(shape, dtype)
        return K.minimum(0.0, w_)


def relu_(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, slope=V_slope.name, **kwargs):

    if mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
        raise ValueError("unknown mode {}".format(mode))

    z_value = K.cast(0.0, K.floatx())
    o_value = K.cast(1.0, K.floatx())

    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensors]
    elif mode == F_IBP.name:
        # y, x_0, u_c, l_c = x[:4]
        u_c, l_c = x[:nb_tensors]
    elif mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = x[:6]
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensors]

    if mode == F_FORWARD.name:
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    # if mode == F_HYBRID.name:
    #    upper = K.minimum(u_c,upper)
    #    lower = K.maximum(l_c, lower)
    if mode in [F_IBP.name, F_HYBRID.name]:
        upper = u_c
        lower = l_c

    if dc_decomp:
        h, g = x[-2:]
        h_ = K.maximum(h, -g)
        g_ = g
        index_dead = -K.clip(K.sign(upper) - o_value, -o_value, z_value)  # =1 if inactive state
        index_linear = K.clip(K.sign(lower) + o_value, z_value, o_value)  # 1 if linear state

        h_ = (o_value - index_dead) * h_
        g_ = (o_value - index_dead) * g_
        h_ = (o_value - index_linear) * h_ + index_linear * h
        g_ = (o_value - index_linear) * g_ + index_linear * g

    u_c_ = K.relu(upper)
    l_c_ = K.relu(lower)

    if mode in [F_FORWARD.name, F_HYBRID.name]:
        w_u_, b_u_, w_l_, b_l_ = get_linear_hull_relu(upper, lower, slope, **kwargs)
        b_u_ = w_u_ * b_u + b_u_
        b_l_ = w_l_ * b_l + b_l_
        w_u_ = K.expand_dims(w_u_, 1) * w_u
        w_l_ = K.expand_dims(w_l_, 1) * w_l

    output = []
    if mode == F_IBP.name:
        output += [u_c_, l_c_]
    if mode == F_FORWARD.name:
        output += [x_0, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        output += [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    if dc_decomp:
        output += [h_, g_]
    return output


def softplus_(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, slope=V_slope.name, **kwargs):

    if mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
        raise ValueError("unknown mode {}".format(mode))

    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensors]
    elif mode == F_IBP.name:
        # y, x_0, u_c, l_c = x[:4]
        u_c, l_c = x[:nb_tensors]
    elif mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = x[:6]
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensors]

    if dc_decomp:
        h, g = x[-2:]
        h_ = K.maximum(h, -g)
        g_ = g

    if mode == F_FORWARD.name:
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    if mode == F_HYBRID.name:
        upper = u_c
        lower = l_c
    if mode == F_IBP.name:
        upper = u_c
        lower = l_c

    u_c_ = K.softplus(upper)
    l_c_ = K.softplus(lower)

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        w_u_, b_u_, w_l_, b_l_ = get_linear_softplus_hull(upper, lower, slope, **kwargs)
        b_u_ = w_u_ * b_u + b_u_
        b_l_ = w_l_ * b_l + b_l_
        w_u_ = K.expand_dims(w_u_, 1) * w_u
        w_l_ = K.expand_dims(w_l_, 1) * w_l

    if mode == F_IBP.name:
        return [u_c_, l_c_]
    if mode == F_FORWARD.name:
        return [x_0, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def substract(inputs_0, inputs_1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of inputs_0-inputs_1

    :param inputs_0: tensor
    :param inputs_1: tensor
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :return: inputs_0 - inputs_1
    """

    inputs_1_ = minus(inputs_1, mode=mode, dc_decomp=dc_decomp)
    return add(inputs_0, inputs_1_, dc_decomp=dc_decomp, mode=mode, convex_domain=convex_domain)


def add(inputs_0, inputs_1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of inputs_0+inputs_1

    :param inputs_0: tensor
    :param inputs_1: tensor
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :return: inputs_0 + inputs_1
    """
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if dc_decomp:
        h_0, g_0 = inputs_0[-2:]
        h_1, g_1 = inputs_1[-2:]
        h_ = h_0 + h_1
        g_ = g_0 + g_1

    if mode == F_HYBRID.name:
        # y_0, x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = inputs_0[:8]
        # y_1, _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = inputs_1[:8]
        x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = inputs_0[:nb_tensor]
        _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = inputs_1[:nb_tensor]
    if mode == F_IBP.name:
        # y_0, x_0, u_c_0, l_c_0 = inputs_0[:4]
        # y_1, _, u_c_1, l_c_1 = inputs_1[:4]
        u_c_0, l_c_0 = inputs_0[:nb_tensor]
        u_c_1, l_c_1 = inputs_1[:nb_tensor]
    if mode == F_FORWARD.name:
        # y_0, x_0, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0[:6]
        # y_1, _, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1[:6]
        x_0, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0[:nb_tensor]
        _, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1[:nb_tensor]

    if mode in [F_HYBRID.name, F_IBP.name]:
        u_c_ = u_c_0 + u_c_1
        l_c_ = l_c_0 + l_c_1
    if mode in [F_HYBRID.name, F_FORWARD.name]:

        w_u_ = w_u_0 + w_u_1
        w_l_ = w_l_0 + w_l_1

        b_u_ = b_u_0 + b_u_1
        b_l_ = b_l_0 + b_l_1

    if mode == F_HYBRID.name:

        upper_ = get_upper(x_0, w_u_, b_u_, convex_domain)  # we can see an improvement
        u_c_ = K.minimum(upper_, u_c_)

        lower_ = get_lower(x_0, w_l_, b_l_, convex_domain)  # we can see an improvement
        l_c_ = K.maximum(lower_, l_c_)

    # y_ = y_0 + y_1

    if mode == F_HYBRID.name:
        # output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    if mode == F_IBP.name:
        # output = [y_, x_0, u_c_, l_c_]
        output = [u_c_, l_c_]
    if mode == F_FORWARD.name:
        # output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
        output = [x_0, w_u_, b_u_, w_l_, b_l_]

    if dc_decomp:
        output += [h_, g_]

    return output

def sum(x, axis=-1, dc_decomp=False, mode=F_HYBRID.name, **kwargs):

    if dc_decomp:
        raise NotImplementedError()

    if mode == F_IBP.name:
        return [K.sum(x[0], axis=axis), K.sum(x[1], axis=axis)]
    if mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x
        u_c_ = K.sum(u_c, axis=axis)
        l_c_ = K.sum(l_c, axis=axis)

    axis_w = -1
    if axis!=-1:
        axis_w = axis+1
    w_u_ = K.sum(w_u, axis=axis_w)
    w_l_ = K.sum(w_l, axis=axis_w)
    b_u_ = K.sum(b_u, axis=axis)
    b_l_ = K.sum(b_l, axis=axis)

    if mode == F_FORWARD.name:
        return [x_0, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

def frac_pos(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):

    if dc_decomp:
        raise NotImplementedError()
    # frac_pos is convex for positive values
    if mode ==F_IBP.name:
        u_c, l_c = x
        u_c_ = 1. / l_c
        l_c_ = 1. / u_c
        return [u_c_, l_c_]
    if mode == F_FORWARD.name:
        x_0, w_u,b_u, w_l, b_l = x
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c = get_lower(x_0, w_u, b_u, convex_domain=convex_domain)
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x
        u_c_ = 1./l_c
        l_c_ = 1./u_c

        w_u_0 = (u_c_-l_c_)/K.maximum(u_c-l_c, K.epsilon())
        b_u_0 = l_c_ - w_u_0*l_c

        y = (u_c+l_c)/2.
        b_l_0 = 2./y
        w_l_0 = -1/y**2

        w_u_ = w_u_0[:, None] * w_l
        b_u_ = b_u_0 * b_l + b_u_0
        w_l_ = w_l_0[:, None] * w_u
        b_l_ = b_l_0 * b_u + b_l_0

        if mode == F_FORWARD.name:
            return [x_0, w_u_, b_u_, w_l_, b_l_]
        if mode == F_HYBRID.name:
            return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]



def maximum(inputs_0, inputs_1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, finetune=False, **kwargs):
    """
    LiRPA implementation of element-wise max

    :param inputs_0: list of tensors
    :param inputs_1: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :return: maximum(inputs_0, inputs_1)
    """
    output_0 = substract(inputs_1, inputs_0, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    if finetune:
        finetune=kwargs['finetune_params']
        output_1 = relu_(
            output_0,
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            mode=mode,
            finetune=finetune)
    else:
        output_1 = relu_(
            output_0,
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            mode=mode)

    return add(
        output_1,
        inputs_0,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        mode=mode,
    )


def minimum(inputs_0, inputs_1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, finetune=False, **kwargs):
    """
    LiRPA implementation of element-wise min

    :param inputs_0:
    :param inputs_1:
    :param dc_decomp:
    :param convex_domain:
    :param mode:
    :return:
    """

    return minus(
        maximum(
            minus(inputs_0, dc_decomp=dc_decomp, mode=mode),
            minus(inputs_1, dc_decomp=dc_decomp, mode=mode),
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            mode=mode,
            finetune=finetune,
            **kwargs
        ),
        dc_decomp=dc_decomp,
        mode=mode,
    )


# convex hull of the maximum between two functions
def max_(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, axis=-1, finetune=False, **kwargs):
    """
    LiRPA implementation of max(x, axis)

    :param x: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :param axis: axis to perform the maximum
    :return: max operation  along an axis
    """

    if dc_decomp:
        h, g = x[-2:]

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensor]
    if mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = x[:6]
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensor]
    if mode == F_IBP.name:
        # y, x_0, u_c, l_c = x[:4]
        u_c, l_c = x[:nb_tensor]

    if mode == F_IBP.name and not dc_decomp:
        u_c_ = K.max(u_c, axis=axis)
        l_c_ = K.max(l_c, axis=axis)

        return [u_c_, l_c_]

    input_shape = K.int_shape(x[-1])
    max_dim = input_shape[axis]

    # do some transpose so that the last axis is also at the end

    if dc_decomp:
        h_list = tf.split(h, max_dim, axis)
        g_list = tf.split(g, max_dim, axis)
        h_tmp = h_list[0] + 0 * (h_list[0])
        g_tmp = g_list[0] + 0 * (g_list[0])

    # y_list = tf.split(y, max_dim, axis)
    # y_tmp = y_list[0] + 0 * (y_list[0])

    if mode in [F_HYBRID.name, F_IBP.name]:

        u_c_list = tf.split(u_c, max_dim, axis)
        l_c_list = tf.split(l_c, max_dim, axis)
        u_c_tmp = u_c_list[0] + 0 * (u_c_list[0])
        l_c_tmp = l_c_list[0] + 0 * (l_c_list[0])

    if mode in [F_HYBRID.name, F_FORWARD.name]:

        b_u_list = tf.split(b_u, max_dim, axis)
        b_l_list = tf.split(b_l, max_dim, axis)
        b_u_tmp = b_u_list[0] + 0 * (b_u_list[0])
        b_l_tmp = b_l_list[0] + 0 * (b_l_list[0])

        if axis == -1:
            w_u_list = tf.split(w_u, max_dim, axis)
            w_l_list = tf.split(w_l, max_dim, axis)
        else:
            w_u_list = tf.split(w_u, max_dim, axis + 1)
            w_l_list = tf.split(w_l, max_dim, axis + 1)
        w_u_tmp = w_u_list[0] + 0 * (w_u_list[0])
        w_l_tmp = w_l_list[0] + 0 * (w_l_list[0])

        if finetune:
            key = [e for e in kwargs.keys()][0]
            params = kwargs[key][0]
            params_ = [e[0] for e in tf.split(params[None], max_dim, axis)]

    output_tmp = []
    if mode == F_HYBRID.name:
        # output_tmp = [y_tmp,x_0,u_c_tmp,w_u_tmp,b_u_tmp,l_c_tmp,w_l_tmp,b_l_tmp,]
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
            # output_i = [y_list[i], x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]
            output_i = [x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]
            if finetune:
                output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune, finetune_params=params_[i])
            else:
                output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    if mode == F_IBP.name:
        # output_tmp = [y_tmp,x_0,u_c_tmp,l_c_tmp,]
        output_tmp = [
            u_c_tmp,
            l_c_tmp,
        ]

        for i in range(1, max_dim):
            # output_i = [y_list[i], x_0, u_c_list[i], l_c_list[i]]
            output_i = [u_c_list[i], l_c_list[i]]
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune)

    if mode == F_FORWARD.name:
        # output_tmp = [y_tmp,x_0,w_u_tmp,b_u_tmp,w_l_tmp,b_l_tmp,]
        output_tmp = [
            x_0,
            w_u_tmp,
            b_u_tmp,
            w_l_tmp,
            b_l_tmp,
        ]

        for i in range(1, max_dim):
            # output_i = [y_list[i], x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
            output_i = [x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
            if finetune:
                output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune, finetune_params=params_[i])
            else:
                output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    # reduce the dimension
    if mode == F_IBP.name:
        # _, _, u_c_, l_c_ = output_tmp[:4]
        u_c_, l_c_ = output_tmp[:nb_tensor]
    if mode == F_FORWARD.name:
        # _, _, w_u_, b_u_, w_l_, b_l_ = output_tmp[:6]
        _, w_u_, b_u_, w_l_, b_l_ = output_tmp[:nb_tensor]
    if mode == F_HYBRID.name:
        # _, _, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_tmp[:8]
        _, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_tmp[:nb_tensor]

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = K.squeeze(u_c_, axis)
        l_c_ = K.squeeze(l_c_, axis)
    if mode in [F_HYBRID.name, F_FORWARD.name]:
        b_u_ = K.squeeze(b_u_, axis)
        b_l_ = K.squeeze(b_l_, axis)
        if axis == -1:
            w_u_ = K.squeeze(w_u_, axis)
            w_l_ = K.squeeze(w_l_, axis)
        else:
            w_u_ = K.squeeze(w_u_, axis + 1)
            w_l_ = K.squeeze(w_l_, axis + 1)

    if dc_decomp:
        g_ = K.sum(g, axis=axis)
        h_ = K.max(h + g, axis=axis) - g_

    if mode == F_HYBRID.name:

        upper_ = get_upper(x_0, w_u_, b_u_, convex_domain)
        u_c_ = K.minimum(upper_, u_c_)

        lower_ = get_lower(x_0, w_l_, b_l_, convex_domain)
        l_c_ = K.maximum(lower_, l_c_)

    # y_ = K.max(y, axis=axis)

    if mode == F_HYBRID.name:
        # output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    if mode == F_IBP.name:
        # output = [y_, x_0, u_c_, l_c_]
        output = [u_c_, l_c_]
    if mode == F_FORWARD.name:
        # output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
        output = [x_0, w_u_, b_u_, w_l_, b_l_]

    if dc_decomp:
        output += [h_, g_]

    return output


def softmax_to_linear(model):
    """
    linearize the softmax layer for verification
    :param model: Keras Model
    :return: model without the softmax
    """
    layer = model.layers[-1]
    # check that layer is not an instance of the object Softmax
    if isinstance(layer, keras.layers.Softmax):
        model_normalize = keras.models.Model(model.get_input_at(0), keras.layers.Activation("linear")(layer.input))

        return model_normalize, True

    if hasattr(layer, "activation"):
        if not layer.get_config()["activation"] == "softmax":
            return model, False
        layer.get_config()["activation"] = "linear"
        layer.activation = keras.activations.get("linear")
        return model, True

    return model, False


def linear_to_softmax(model):

    model.layers[-1].activation = keras.activations.get("softmax")
    return model


def minus(inputs, mode=F_HYBRID.name, dc_decomp=False):
    """
    LiRPA implementation of minus(x)=-x.
    :param inputs:
    :param mode:
    :return:
    """
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        # y, x, u, l = inputs[:4]
        u, l = inputs[:nb_tensor]
    if mode == F_FORWARD.name:
        # y, x, w_u, b_u, w_l, b_l = inputs[:6]
        x, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    if mode == F_HYBRID.name:
        # y, x, u, w_u, b_u, l, w_l, b_l = inputs[:8]
        x, u, w_u, b_u, l, w_l, b_l = inputs[:nb_tensor]

    if dc_decomp:
        h, g = inputs[-2:]

    # y_ = -y

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_ = -l
        l_ = -u

    if mode in [F_FORWARD.name, F_HYBRID.name]:
        w_u_ = -w_l
        b_u_ = -b_l
        w_l_ = -w_u
        b_l_ = -b_u

    # output = [y_, x]
    if mode == F_IBP.name:
        output = [u_, l_]
    if mode == F_FORWARD.name:
        output = [x, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        output = [x, u_, w_u_, b_u_, l_, w_l_, b_l_]

    if dc_decomp:
        output += [-g, -h]

    return output


def multiply_old(inputs_0, inputs_1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of (element-wise) multiply(x,y)=-x*y.

    :param inputs_0: list of tensors
    :param inputs_1: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: maximum(inputs_0, inputs_1)
    """

    if dc_decomp:
        raise NotImplementedError()

    nb_tensor = StaticVariables(dc_decomp, mode=mode).nb_tensors

    if mode == F_IBP.name:
        # y0, x, u0, l0 = inputs_0[:4]
        # y1, _, u1, l1 = inputs_1[:4]
        u0, l0 = inputs_0[:nb_tensor]
        u1, l1 = inputs_1[:nb_tensor]
    if mode == F_FORWARD.name:
        # y0, x, w_u0, b_u0, w_l0, b_l0 = inputs_0[:6]
        # y1, _, w_u1, b_u1, w_l1, b_l1 = inputs_1[:6]
        x, w_u0, b_u0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, w_u1, b_u1, w_l1, b_l1 = inputs_1[:nb_tensor]
        u0 = get_upper(x, w_u0, b_u0, convex_domain=convex_domain)
        l0 = get_lower(x, w_l0, b_l0, convex_domain=convex_domain)
        u1 = get_upper(x, w_u1, b_u1, convex_domain=convex_domain)
        l1 = get_lower(x, w_l1, b_l1, convex_domain=convex_domain)
    if mode == F_HYBRID.name:
        # y0, x, u0, w_u0, b_u0, l0, w_l0, b_l0 = inputs_0[:8]
        # y1, _, u1, w_u1, b_u1, l1, w_l1, b_l1 = inputs_1[:8]
        x, u0, w_u0, b_u0, l0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, u1, w_u1, b_u1, l1, w_l1, b_l1 = inputs_1[:nb_tensor]

    # using McCormick's inequalities to derive bounds
    # xy<= x_u*y + x*y_L - xU*y_L
    # xy<= x*y_u + x_L*y - x_L*y_U

    # xy >=x_L*y + x*y_L -x_L*y_L
    # xy >= x_U*y + x*y_U - x_U*y_U
    if mode in [F_IBP.name, F_HYBRID.name]:


        upper_0 = K.relu(u0) * u1 - K.relu(-u0) * l1 + K.relu(l1) * u0 - K.relu(-l1) * l0 - u0 * l1
        upper_1 = K.relu(u1) * u0 - K.relu(-u1) * l0 + K.relu(l0) * u1 - K.relu(-l0) * l1 - u1 * l0

        lower_0 = K.relu(l0) * l1 - K.relu(-l0) * u1 + K.relu(l1) * l0 - K.relu(-l1) * u0 - l0 * l1
        lower_1 = K.relu(u1) * l0 - K.relu(-u1) * u0 + K.relu(u0) * l1 - K.relu(-u0) * u1 - u1 * u0

    if mode in [F_FORWARD.name, F_HYBRID.name]:
        w_0_u = (
            K.relu(u0)[:, None] * w_u1
            - K.relu(-u0)[:, None] * w_l1
            + K.relu(l1)[:, None] * w_u0
            - K.relu(-l1)[:, None] * w_l0
        )
        w_1_u = (
            K.relu(u1)[:, None] * w_u0
            - K.relu(-u1)[:, None] * w_l0
            + K.relu(l0)[:, None] * w_u1
            - K.relu(-l0)[:, None] * w_l1
        )

        w_0_l = (
            K.relu(l0)[:, None] * w_l1
            - K.relu(-l0)[:, None] * w_u1
            + K.relu(l1)[:, None] * w_l0
            - K.relu(-l1)[:, None] * w_u0
        )
        w_1_l = (
            K.relu(u1)[:, None] * w_l0
            - K.relu(-u1)[:, None] * w_u0
            + K.relu(u0)[:, None] * w_l1
            - K.relu(-u0)[:, None] * w_u1
        )

        b_u_0 = K.relu(u0) * b_u1 - K.relu(-u0) * b_l1 + K.relu(l1) * b_u0 - K.relu(-l1) * b_l0 - u0 * l1
        b_u_1 = K.relu(u1) * b_u0 - K.relu(-u1) * b_l0 + K.relu(l0) * b_u1 - K.relu(-l0) * b_l1 - u1 * l0

        b_l_0 = K.relu(l0) * b_l1 - K.relu(-l0) * b_u1 + K.relu(l1) * b_l0 - K.relu(-l1) * b_u0 - l0 * l1
        b_l_1 = K.relu(u1) * b_l0 - K.relu(-u1) * b_u0 + K.relu(u0) * b_l1 - K.relu(-u0) * b_u1 - u1 * u0

    # if mode == F_IBP.name:
    #    inputs_0_ = [y0*y1, x, upper_0, lower_0]
    #    inputs_1_ = [y0*y1, x, upper_1, lower_1]

    if mode == F_HYBRID.name:
        # inputs_0_ = [y0 * y1, x, upper_0, w_0_u, b_u_0, lower_0, w_0_l, b_l_0]
        # inputs_1_ = [y0 * y1, x, upper_1, w_1_u, b_u_1, lower_1, w_1_l, b_l_1]
        inputs_0_ = [x, upper_0, w_0_u, b_u_0, lower_0, w_0_l, b_l_0]
        inputs_1_ = [x, upper_1, w_1_u, b_u_1, lower_1, w_1_l, b_l_1]

    if mode == F_FORWARD.name:
        # inputs_0_ = [y0 * y1, x, w_0_u, b_u_0, w_0_l, b_l_0]
        # inputs_1_ = [y0 * y1, x, w_1_u, b_u_1, w_1_l, b_l_1]
        inputs_0_ = [x, w_0_u, b_u_0, w_0_l, b_l_0]
        inputs_1_ = [x, w_1_u, b_u_1, w_1_l, b_l_1]

    if mode == F_IBP.name:
        # output = [y0 * y1, x, K.minimum(upper_0, upper_1), K.maximum(lower_0, lower_1)]
        output = [K.minimum(upper_0, upper_1), K.maximum(lower_0, lower_1)]
    else:
        output_upper = minimum(inputs_0_, inputs_1_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
        output_lower = maximum(inputs_0_, inputs_1_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)

        if mode == F_FORWARD.name:
            # _, _, w_u_, b_u_, _, _ = output_upper
            # _, _, _, _, w_l_, b_l_ = output_upper

            _, w_u_, b_u_, _, _ = output_upper
            _, _, _, w_l_, b_l_ = output_lower

            # output = [y0 * y1, x, w_u_, b_u_, w_l_, b_l_]
            output = [x, w_u_, b_u_, w_l_, b_l_]

        if mode == F_HYBRID.name:
            # _, _, u_, w_u_, b_u_, _, _, _ = output_upper
            # _, _, _, _, _, l_, w_l_, b_l_ = output_upper
            _, u_, w_u_, b_u_, _, _, _ = output_upper
            _, _, _, _, l_, w_l_, b_l_ = output_upper

            # output = [y0 * y1, x, u_, w_u_, b_u_, l_, w_l_, b_l_]
            output = [x, u_, w_u_, b_u_, l_, w_l_, b_l_]

    return output


def multiply(inputs_0, inputs_1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of (element-wise) multiply(x,y)=-x*y.

    :param inputs_0: list of tensors
    :param inputs_1: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: maximum(inputs_0, inputs_1)
    """

    if dc_decomp:
        raise NotImplementedError()

    nb_tensor = StaticVariables(dc_decomp, mode=mode).nb_tensors

    if mode == F_IBP.name:
        # y0, x, u0, l0 = inputs_0[:4]
        # y1, _, u1, l1 = inputs_1[:4]
        u0, l0 = inputs_0[:nb_tensor]
        u1, l1 = inputs_1[:nb_tensor]
    if mode == F_FORWARD.name:
        # y0, x, w_u0, b_u0, w_l0, b_l0 = inputs_0[:6]
        # y1, _, w_u1, b_u1, w_l1, b_l1 = inputs_1[:6]
        x, w_u0, b_u0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, w_u1, b_u1, w_l1, b_l1 = inputs_1[:nb_tensor]
        u0 = get_upper(x, w_u0, b_u0, convex_domain=convex_domain)
        l0 = get_lower(x, w_l0, b_l0, convex_domain=convex_domain)
        u1 = get_upper(x, w_u1, b_u1, convex_domain=convex_domain)
        l1 = get_lower(x, w_l1, b_l1, convex_domain=convex_domain)
    if mode == F_HYBRID.name:
        # y0, x, u0, w_u0, b_u0, l0, w_l0, b_l0 = inputs_0[:8]
        # y1, _, u1, w_u1, b_u1, l1, w_l1, b_l1 = inputs_1[:8]
        x, u0, w_u0, b_u0, l0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, u1, w_u1, b_u1, l1, w_l1, b_l1 = inputs_1[:nb_tensor]

    # using McCormick's inequalities to derive bounds
    # xy<= x_u*y + x*y_L - xU*y_L
    # xy<= x*y_u + x_L*y - x_L*y_U

    # xy >=x_L*y + x*y_L -x_L*y_L
    # xy >= x_U*y + x*y_U - x_U*y_U

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_0_ = K.maximum(u0*u1, u0*l1) + K.maximum(u0*l1, l0*l1) - u0*l1
        u_c_1_ = K.maximum(u1*u0, u1*l0) + K.maximum(u1*l0, l1*l0) - u1*l0
        u_c_ = K.minimum(u_c_0_, u_c_1_)
        l_c_ = K.minimum(l0*l1, l0*u1) + K.minimum(l0*l1, u0*l1) - l0*l1

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        #xy <= x_u * y + x * y_L - xU * y_L
        cx_u_pos = K.maximum(u0, 0.)
        cx_u_neg = K.minimum(u0, 0.)

        cy_l_pos = K.maximum(l1, 0.)
        cy_l_neg = K.minimum(l1, 0.)
        w_u_ = cx_u_pos[:,None]*w_u1 + cx_u_neg[:,None]*w_l1 + cy_l_pos[:,None]*w_u0 + cy_l_neg[:,None]*w_l0
        b_u_ = cx_u_pos*b_u1 + cx_u_neg*b_l1 + cy_l_pos*b_u0 + cy_l_neg*b_l0 - u0*l1

        # xy >= x_U*y + x*y_U - x_U*y_U
        cy_u_pos = K.maximum(u1, 0.)
        cy_u_neg = K.minimum(u1, 0.)
        cx_l_pos = K.maximum(l0, 0.)
        cx_l_neg = K.minimum(l0, 0.)


        w_l_ = cx_l_pos[:,None]*w_l1 + cx_l_neg[:,None]*w_u1 + cy_l_pos[:,None]*w_l0 + cy_l_neg[:,None]*w_u0
        b_l_ = cx_l_pos*b_l1 + cx_l_neg*b_u1 + cy_l_pos*b_l0 + cy_l_neg*b_u0 - l0*l1

    if mode == F_IBP.name:
        return [u_c_, l_c_]
    if mode == F_FORWARD.name:
        return [x, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        return [x, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

def permute_dimensions(x, axis, mode=F_HYBRID.name, axis_perm=1):
    """
    LiRPA implementation of (element-wise) permute(x,axis)

    :param x: list of input tensors
    :param axis: axis on which we apply the permutation
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param axis_perm: see DecomonPermute operator
    :return:
    """

    if len(x[0].shape) <= 2:
        return x
    index = np.arange(len(x[0].shape))
    index = np.insert(np.delete(index, axis), axis_perm, axis)
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        return [
            # K.permute_dimensions(x[0], index),
            # x[1],
            K.permute_dimensions(x[2], index),
            K.permute_dimensions(x[3], index),
        ]

    index_w = np.arange(len(x[0].shape) + 1)
    index_w = np.insert(np.delete(index_w, axis), axis_perm + 1, axis)

    if mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = x
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensor]
        return [
            # K.permute_dimensions(y, index),
            x_0,
            K.permute_dimensions(w_u, index_w),
            K.permute_dimensions(b_u, index),
            K.permute_dimensions(w_l, index_w),
            K.permute_dimensions(b_l, index),
        ]

    if mode == F_HYBRID.name:
        # y, x_0, u, w_u, b_u, l, w_l, b_l = x
        x_0, u, w_u, b_u, l, w_l, b_l = x

        return [
            # K.permute_dimensions(y, index),
            x_0,
            K.permute_dimensions(u, index),
            K.permute_dimensions(w_u, index_w),
            K.permute_dimensions(b_u, index),
            K.permute_dimensions(l, index),
            K.permute_dimensions(w_l, index_w),
            K.permute_dimensions(b_l, index),
        ]


def broadcast(inputs, n, axis, mode):
    """
    LiRPA implementation of broadcasting

    :param inputs:
    :param n:
    :param axis:
    :param mode:
    :return:
    """
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        # y, x, u, l = inputs
        u, l = inputs[:nb_tensor]
    if mode == F_FORWARD.name:
        # y, x, w_u, b_u, w_l, b_l = inputs
        x, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    if mode == F_HYBRID.name:
        # y, x, u, w_u, b_u, l, w_l, b_l = inputs
        x, u, w_u, b_u, l, w_l, b_l = inputs[:nb_tensor]

    # for _ in range(n):
    # y = K.expand_dims(y, axis)

    if mode in [F_IBP.name, F_HYBRID.name]:
        for _ in range(n):
            u = K.expand_dims(u, axis)
            l = K.expand_dims(l, axis)

    if axis != -1:
        axis_w = axis + 1
    else:
        axis_w = -1

    if mode in [F_FORWARD.name, F_HYBRID.name]:
        for _ in range(n):
            b_u = K.expand_dims(b_u, axis)
            b_l = K.expand_dims(b_l, axis)
            w_u = K.expand_dims(w_u, axis_w)
            w_l = K.expand_dims(w_l, axis_w)

    if mode == F_IBP.name:
        # output = [y, x, u, l]
        output = [u, l]
    if mode == F_FORWARD.name:
        # output = [y, x, w_u, b_u, w_l, b_l]
        output = [x, w_u, b_u, w_l, b_l]
    if mode == F_HYBRID.name:
        # output = [y, x, u, w_u, b_u, l, w_l, b_l]
        output = [x, u, w_u, b_u, l, w_l, b_l]

    return output


def split(input_, axis=-1, mode=F_HYBRID.name):
    """
    LiRPA implementation of split

    :param input_:
    :param axis:
    :param mode:
    :return:
    """

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        # y_, x_, u_, l_ = input_
        u_, l_ = input_[:nb_tensor]

    if mode == F_HYBRID.name:
        # y_, x_, u_, w_u_, b_u_, l_, w_l_, b_l_ = input_
        x_, u_, w_u_, b_u_, l_, w_l_, b_l_ = input_[:nb_tensor]

    if mode == F_FORWARD.name:
        # y_, x_, w_u_, b_u_, w_l_, b_l_ = input_
        x_, w_u_, b_u_, w_l_, b_l_ = input_[:nb_tensor]

    # y_list = tf.split(y_, 1, axis=axis)

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_list = tf.split(u_, 1, axis=axis)
        l_list = tf.split(l_, 1, axis=axis)
        n = len(u_list)

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        b_u_list = tf.split(b_u_, 1, axis=axis)
        b_l_list = tf.split(b_l_, 1, axis=axis)
        n = len(b_u_list)

        if axis != -1:
            axis += 1
        w_u_list = tf.split(w_u_, 1, axis=axis)
        w_l_list = tf.split(w_l_, 1, axis=axis)

    if mode == F_IBP.name:
        # outputs = [[y_list[i], x_, u_list[i], l_list[i]] for i in range(n)]
        outputs = [[u_list[i], l_list[i]] for i in range(n)]

    if mode == F_FORWARD.name:
        # outputs = [[y_list[i], x_, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]
        outputs = [[x_, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]

    if mode == F_HYBRID.name:
        # outputs = [[y_list[i], x_, u_list[i], w_u_list[i], b_u_list[i], l_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]
        outputs = [[x_, u_list[i], w_u_list[i], b_u_list[i], l_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]

    return outputs


def sort(input_, axis=-1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of sort by selection

    :param input_:
    :param axis:
    :param dc_decomp:
    :param convex_domain:
    :param mode:
    :return:
    """

    if dc_decomp:
        raise NotImplementedError()
    # remove grad bounds

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == F_IBP.name:
        # y, x_0, u_c, l_c = input_
        u_c, l_c = input_[:nb_tensor]

    if mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = input_
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = input_[:nb_tensor]

    if mode == F_FORWARD.name:
        # y_, x_0, w_u, b_u, w_l, b_l = input_
        x_0, w_u, b_u, w_l, b_l = input_[:nb_tensor]

    if axis == -1:
        n = input_.shape[-1]
        axis = len(input_.shape) - 1
    else:
        n = input_.shape[axis]

    # y_ = tf.sort(y, axis=axis)

    # what about splitting elements
    op = lambda x: tf.split(x, n, axis=axis)
    # y_list = op(y)

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_list = op(u_c)
        l_c_list = op(l_c)

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        w_u_list = tf.split(w_u, n, axis=axis + 1)
        b_u_list = op(b_u)
        w_l_list = tf.split(w_l, n, axis=axis + 1)
        b_l_list = op(b_l)

    def get_input(mode, i):
        if mode == F_IBP.name:
            # return [y_list[i], x_0, u_c_list[i], l_c_list[i]]
            return [u_c_list[i], l_c_list[i]]
        if mode == F_FORWARD.name:
            # return [y_list[i], x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
            return [x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
        if mode == F_HYBRID.name:
            # return [y_list[i], x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]
            return [x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]

    def set_input(input_, mode, i):
        if mode == F_IBP.name:
            # y_i, _, u_i, l_i = input_
            u_i, l_i = input_
        if mode == F_FORWARD.name:
            # y_i, _, w_u_i, b_u_i, w_l_i, b_l_i = input_
            _, w_u_i, b_u_i, w_l_i, b_l_i = input_
        if mode == F_HYBRID.name:
            # y_i, _, u_i, w_u_i, b_u_i, l_i, w_l_i, b_l_i = input_
            _, u_i, w_u_i, b_u_i, l_i, w_l_i, b_l_i = input_

        # y_list[i] = y_i
        if mode in [F_IBP.name, F_HYBRID.name]:
            u_c_list[i] = u_i
            l_c_list[i] = l_i
        if mode in [F_FORWARD.name, F_HYBRID.name]:
            w_u_list[i] = w_u_i
            w_l_list[i] = w_l_i
            b_u_list[i] = b_u_i
            b_l_list[i] = b_l_i

    # use selection sort
    for i in range(n - 1):
        for j in range(i + 1, n):

            input_i = get_input(mode, i)
            input_j = get_input(mode, j)
            output_a = maximum(input_i, input_j, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp)
            output_b = minimum(input_i, input_j, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp)

            set_input(output_a, mode, j)
            set_input(output_b, mode, i)

    op_ = lambda x: K.concatenate(x, axis)
    # update the inputs
    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = op_(u_c_list)
        l_c_ = op_(l_c_list)
    if mode in [F_FORWARD.name, F_HYBRID.name]:
        w_u_ = K.concatenate(w_u_list, axis + 1)
        w_l_ = K.concatenate(w_l_list, axis + 1)
        b_u_ = op_(b_u_list)
        b_l_ = op_(b_l_list)

    if mode == F_IBP.name:
        # output = [y_, x_0, u_c_, l_c_]
        output = [u_c_, l_c_]
    if mode == F_FORWARD.name:
        # output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
        output = [x_0, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        # output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    return output


def pow(inputs_, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of pow(x )=x**2

    :param inputs_:
    :param dc_decomp:
    :param convex_domain:
    :param mode:
    :return:
    """

    return multiply(inputs_, inputs_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def abs(inputs_, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of |x|

    :param inputs_:
    :param dc_decomp:
    :param convex_domain:
    :param mode:
    :return:
    """

    inputs_0 = relu_(inputs_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    inputs_1 = minus(
        relu_(minus(inputs_, mode=mode), dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode), mode=mode
    )

    return add(inputs_0, inputs_1, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def frac_pos_hull(inputs_, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of 1/x for x>0
    :param inputs_:
    :param dc_decomp:
    :param convex_domain:
    :param mode:
    :return:
    """

    if dc_decomp:
        raise NotImplementedError()
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == F_IBP.name:
        # y, x_0, u_c, l_c = inputs_
        u_c, l_c = inputs_[:nb_tensor]
    if mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = inputs_
        x_0, w_u, b_u, w_l, b_l = inputs_[:nb_tensor]
    if mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs_
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs_[:nb_tensor]

    # y_ = 1 / y
    if mode in [F_FORWARD.name, F_HYBRID.name]:
        u_c_ = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c_ = get_lower(x_0, w_u, b_u, convex_domain=convex_domain)

        if mode == F_FORWARD.name:
            u_c = u_c_
            l_c = l_c_
        else:
            u_c = K.minimum(u_c_, u_c)
            l_c = K.maximum(l_c, l_c_)

    l_c = K.maximum(l_c, 1.0)
    z = (u_c + l_c) / 2.0
    w_l = -1 / K.pow(z)
    b_l = 2 / z
    w_u = (1.0 / u_c - 1.0 / l_c) / (u_c - l_c)
    b_u = 1.0 / u_c - w_u * u_c

    return [w_u, b_u, w_l, b_l]


# convex hull for min
def min_(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, axis=-1, finetune=False, **kwargs):
    """
    LiRPA implementation of min(x, axis=axis)
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param mode:
    :param axis:
    :return:
    """
    # return - max - x
    return minus(
        max_(minus(x, mode=mode), dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode, axis=axis, finetune=finetune, **kwargs), mode=mode
    )


def expand_dims(inputs_, dc_decomp=False, mode=F_HYBRID.name, axis=-1):

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        # y, x_0, u_c, l_c = inputs_[:4]
        u_c, l_c = inputs_[:nb_tensor]
    if mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = inputs_[:6]
        x_0, w_u, b_u, w_l, b_l = inputs_[:nb_tensor]

    if mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs_[:8]
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs_[:nb_tensor]

    if dc_decomp:
        h, g = inputs_[-2:]

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        if axis == -1:
            axis_w = axis
        else:
            axis_w = axis + 1

    op = lambda t: K.expand_dims(t, axis)

    # y_ = op(y)
    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = op(u_c)
        l_c_ = op(l_c)

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        b_u_ = op(b_u)
        b_l_ = op(b_l)
        w_u_ = K.expand_dims(w_u, axis_w)
        w_l_ = K.expand_dims(w_l, axis_w)

    if mode == F_IBP.name:
        # output = [y_, x_0, u_c_, l_c_]
        output = [u_c_, l_c_]

    if mode == F_FORWARD.name:
        # output = [y, x_0, w_u_, b_u_, w_l_, b_l_]
        output = [x_0, w_u_, b_u_, w_l_, b_l_]

    if mode == F_HYBRID.name:
        # output = [y, x_0, u_c_, w_u_, l_c_, w_l_, b_l_]
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    return output



