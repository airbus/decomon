from __future__ import absolute_import
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
import tensorflow.keras as keras
import numpy as np
from .core import Ball, Box, Vertex, F_FORWARD, F_IBP, F_HYBRID
from tensorflow.math import greater_equal
from tensorflow.keras.layers import Flatten


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

    if len(w.shape) == len(b.shape):  # identity function
        return x_max

    # split into positive and negative components
    w_pos = K.maximum(w, 0.0)
    w_neg = K.minimum(w, 0.0)

    x_min_ = x_min + 0 * x_min
    x_max_ = x_max + 0 * x_max

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

    w_pos = K.maximum(w, 0.0)
    w_neg = K.minimum(w, 0.0)

    x_min_ = x_min + 0 * x_min
    x_max_ = x_max + 0 * x_max

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


def relu_(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, slope=V_slope.name):
    """
    LiRPA implementation of relu(x)=max(x, 0)

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: the updated list of tensors
    """

    if mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
        raise ValueError("unknown mode {}".format(mode))

    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
    elif mode == F_IBP.name:
        y, x_0, u_c, l_c = x[:4]
    elif mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = x[:6]

    if dc_decomp:
        h, g = x[-2:]
        h_ = K.maximum(h, -g)
        g_ = g

    # compute upper and lower bounds
    # and keep the minimum between the constant and the computed upper bounds
    # and keep the maximum between the constant and the computer lower bounds
    if mode == F_FORWARD.name:
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    if mode == F_HYBRID.name:
        upper = u_c
        lower = l_c
    if mode == F_IBP.name:
        upper = u_c
        lower = l_c

    # check inactive relu state: u<=0
    index_dead = -K.clip(K.sign(upper) - 1, -1, 0)
    index_linear = K.clip(K.sign(lower) + 1, 0, 1)

    # update the new upper and lower constant bounds after relu
    if mode in [F_HYBRID.name, F_IBP.name]:
        u_c_ = K.maximum(0.0, upper)
        l_c_ = K.maximum(0.0, lower)

    if dc_decomp:
        h_ = (1 - index_dead) * h_
        g_ = (1 - index_dead) * g_

    if mode in [F_HYBRID.name, F_FORWARD.name]:

        # in case upper=lower, this cases are
        # considered with index_dead and index_linear
        alpha = upper / K.maximum(K.epsilon(), upper - lower)
        # scaling factor for the upper bound on the relu
        # see README

        w_u_ = K.expand_dims(alpha, 1) * w_u
        b_u_ = alpha * (b_u - lower)

        if slope == V_slope.name:

            # 1 if upper<=-lower else 0
            index_a = -K.clip(K.sign(upper + lower) - 1, -1.0, 0.0)

            # 1 if upper>-lower else 0
            index_b = 1.0 - index_a
            w_l_ = K.expand_dims(index_b, 1) * w_l
            b_l_ = index_b * b_l

        if slope == Z_slope.name:
            w_l_ = 0 * w_l
            b_l_ = 0 * b_l

        if slope == O_slope.name:

            w_l_ = w_l
            b_l_ = b_l

        if slope == S_slope.name:
            w_l_ = K.expand_dims(alpha, 1) * w_l
            b_l_ = alpha * b_l

        # set everything to the initial state if relu_ is linear
        index_linear_b = 1.0 - index_linear
        index_linear_w_0 = K.expand_dims(index_linear, 1)
        index_linear_w_1 = 1.0 - index_linear_w_0

        w_u_ = index_linear_w_1 * w_u_ + index_linear_w_0 * w_u
        w_l_ = index_linear_w_1 * w_l_ + index_linear_w_0 * w_l
        b_u_ = index_linear_b * b_u_ + index_linear * b_u
        b_l_ = index_linear_b * b_l_ + index_linear * b_l

        # set everything to zero if the relu is inactive
        index_dead_b = 1 - index_dead
        index_dead_w = K.expand_dims(index_dead_b, 1)

        w_u_ = index_dead_w * w_u_
        b_u_ = index_dead_b * b_u_

        w_l_ = index_dead_w * w_l_
        b_l_ = index_dead_b * b_l_

    y_ = K.relu(y)

    if mode == F_HYBRID.name:
        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    elif mode == F_IBP.name:
        output = [y_, x_0, u_c_, l_c_]
    elif mode == F_FORWARD.name:
        output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]

    if dc_decomp:
        return output + [h_, g_]
    else:
        return output


def softplus_(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, slope=V_slope.name):
    """
    LiRPA implementation of relu(x)=max(x, 0)

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: the updated list of tensors
    """

    if mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
        raise ValueError("unknown mode {}".format(mode))

    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
    elif mode == F_IBP.name:
        y, x_0, u_c, l_c = x[:4]
    elif mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = x[:6]

    if dc_decomp:
        raise NotImplementedError()

    # compute upper and lower bounds
    # and keep the minimum between the constant and the computed upper bounds
    # and keep the maximum between the constant and the computer lower bounds
    if mode == F_FORWARD.name:
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    if mode == F_HYBRID.name:
        upper = u_c
        lower = l_c
    if mode == F_IBP.name:
        upper = u_c
        lower = l_c

    # check inactive relu state: u<=0

    # update the new upper and lower constant bounds after relu
    u_c_ = K.softplus(upper)
    l_c_ = K.softplus(lower)

    if mode in [F_HYBRID.name, F_FORWARD.name]:

        # in case upper=lower, this cases are
        # considered with index_dead and index_linear
        alpha = (u_c_ - l_c_) / K.maximum(K.epsilon(), (upper - lower))
        w_u_ = K.expand_dims(alpha, 1) * w_u
        b_u_ = alpha * (b_u - lower) + l_c_

        if slope == V_slope.name:

            # 1 if upper<=-lower else 0
            index_a = -K.clip(K.sign(upper + lower) - 1, -1.0, 0.0)

            # 1 if upper>-lower else 0
            index_b = 1.0 - index_a
            w_l_ = K.expand_dims(index_b, 1) * w_l
            b_l_ = index_b * b_l

        if slope == Z_slope.name:
            w_l_ = 0 * w_l
            b_l_ = 0 * b_l

        if slope == O_slope.name:

            w_l_ = w_l
            b_l_ = b_l

        if slope == S_slope.name:
            w_l_ = K.expand_dims(alpha, 1) * w_l
            b_l_ = alpha * b_l

    y_ = K.softplus(y)

    if mode == F_HYBRID.name:
        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    elif mode == F_IBP.name:
        output = [y_, x_0, u_c_, l_c_]
    elif mode == F_FORWARD.name:
        output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]

    if dc_decomp:
        return output
    else:
        return output


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

    if dc_decomp:
        h_0, g_0 = inputs_0[-2:]
        h_1, g_1 = inputs_1[-2:]
        h_ = h_0 + h_1
        g_ = g_0 + g_1

    if mode == F_HYBRID.name:
        y_0, x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = inputs_0[:8]
        y_1, _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = inputs_1[:8]
    if mode == F_IBP.name:
        y_0, x_0, u_c_0, l_c_0 = inputs_0[:4]
        y_1, _, u_c_1, l_c_1 = inputs_1[:4]
    if mode == F_FORWARD.name:
        y_0, x_0, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0[:6]
        y_1, _, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1[:6]

    if mode in [F_HYBRID.name, F_IBP.name]:
        u_c_ = u_c_0 + u_c_1
        l_c_ = l_c_0 + l_c_1
    if mode in [F_HYBRID.name, F_FORWARD.name]:

        w_u_ = w_u_0 + w_u_1
        w_l_ = w_l_0 + w_l_1

        b_u_ = b_u_0 + b_u_1
        b_l_ = b_l_0 + b_l_1

    if mode == F_HYBRID.name:

        upper_ = get_upper(x_0, w_u_, b_u_, convex_domain)
        u_c_ = K.minimum(upper_, u_c_)

        lower_ = get_lower(x_0, w_l_, b_l_, convex_domain)
        l_c_ = K.maximum(lower_, l_c_)

    y_ = y_0 + y_1

    if mode == F_HYBRID.name:
        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    if mode == F_IBP.name:
        output = [y_, x_0, u_c_, l_c_]
    if mode == F_FORWARD.name:
        output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]

    if dc_decomp:
        output += [h_, g_]

    return output


def maximum(inputs_0, inputs_1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
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


def minimum(inputs_0, inputs_1, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name):
    """
    LiRPA implementation of element-wise min

    :param inputs_0:
    :param inputs_1:
    :param dc_decomp:
    :param convex_domain:
    :param mode:
    :return:
    """

    return minus(maximum(minus(inputs_0), minus(inputs_1), dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode))


# convex hull of the maximum between two functions
def max_(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, axis=-1):
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

    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
    if mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = x[:6]
    if mode == F_IBP.name:
        y, x_0, u_c, l_c = x[:4]

    input_shape = K.int_shape(y)
    max_dim = input_shape[axis]

    # do some transpose so that the last axis is also at the end

    if dc_decomp:
        h_list = tf.split(h, max_dim, axis)
        g_list = tf.split(g, max_dim, axis)
        h_tmp = h_list[0] + 0 * (h_list[0])
        g_tmp = g_list[0] + 0 * (g_list[0])

    y_list = tf.split(y, max_dim, axis)
    y_tmp = y_list[0] + 0 * (y_list[0])

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

    output_tmp = []
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
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    # reduce the dimension
    if mode == F_IBP.name:
        _, _, u_c_, l_c_ = output_tmp[:4]
    if mode == F_FORWARD.name:
        _, _, w_u_, b_u_, w_l_, b_l_ = output_tmp[:6]
    if mode == F_HYBRID.name:
        _, _, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_tmp[:8]

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

    y_ = K.max(y, axis=axis)

    if mode == F_HYBRID.name:
        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    if mode == F_IBP.name:
        output = [y_, x_0, u_c_, l_c_]
    if mode == F_FORWARD.name:
        output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]

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

        return model_normalize

    if hasattr(layer, "activation"):
        if not layer.get_config()["activation"] == "softmax":
            return model
        layer.get_config()["activation"] = "linear"
        layer.activation = keras.activations.get("linear")

    return model


# linear hull for activation function
def relu_prime(x):
    """
    Derivative of relu
    :param x:
    :return:
    """

    return K.clip(K.sign(x), 0, 1)


def sigmoid_prime(x):
    """
    Derivative of sigmoid
    :param x:
    :return:
    """

    s_x = K.sigmoid(x)
    return s_x * (1 - s_x)


def tanh_prime(x):
    """
    Derivative of tanh
    :param x:
    :return:
    """

    s_x = K.tanh(x)
    return 1 - K.pow(s_x, 2)


def softsign_prime(x):
    """
    Derivative of softsign
    :param x:
    :return:
    """

    return 1.0 / K.pow(1.0 + K.abs(x), 2)


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

    # step1: find t
    u_c_flat_ = K.expand_dims(u_c_flat, -1)  # (None, n , 1)
    l_c_flat_ = K.expand_dims(l_c_flat, -1)  # (None, n,  1)
    t_ = np.linspace(0, 1.0, 100)[None, None, :] * u_c_flat_  # (None, n , 100)

    s_p_t_ = f_prime(t_)  # (None, n, 100)
    s_t_ = func(t_)  # (None, n, 100)

    score = K.abs(s_p_t_ - (s_t_ - K.expand_dims(s_l, -1)) / (t_ - l_c_flat_))  # (None, n, 100)
    index_ = K.argmin(score, -1)  # (None, n)

    threshold = K.min(score, -1)  # (None, n)
    index_t = K.cast(
        K.switch(K.greater(threshold, 0.0 * threshold), index_, K.clip(index_ - 1, 0, 100)), K.floatx()
    )  # (None, n)
    t_value = K.sum(
        K.switch(
            K.equal(1.0 * np.arange(0, 100)[None, None, :] + 0 * u_c_flat_, K.expand_dims(index_t, -1) + 0 * u_c_flat_),
            t_,
            0 * t_,
        ),
        -1,
    )  # (None, n)

    s_t = func(t_value)  # (None, n)
    w_u = (s_t - s_l) / K.maximum(K.epsilon(), t_value - l_c_flat)  # (None, n)
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

    # step1: find t
    u_c_flat_ = K.expand_dims(u_c_flat, -1)  # (None, n , 1)
    l_c_flat_ = K.expand_dims(l_c_flat, -1)  # (None, n,  1)
    t_ = np.linspace(0, 1.0, 100)[None, None, :] * l_c_flat_  # (None, n , 100)

    s_p_t_ = f_prime(t_)  # (None, n, 100)
    s_t_ = func(t_)  # (None, n, 100)

    score = K.abs(s_p_t_ - (K.expand_dims(s_u, -1) - s_t_) / (u_c_flat_ - t_))  # (None, n, 100)
    index_ = K.argmin(score, -1)  # (None, n)

    threshold = K.min(score, -1)
    index_t = K.cast(
        K.switch(K.greater(threshold, 0.0 * threshold), index_, K.clip(index_ + 1, 0, 100)), K.floatx()
    )  # (None, n)
    t_value = K.sum(
        K.switch(
            K.equal(1.0 * np.arange(0, 100)[None, None, :] + 0 * u_c_flat_, K.expand_dims(index_t, -1) + 0 * u_c_flat_),
            t_,
            0 * t_,
        ),
        -1,
    )

    s_t = func(t_value)  # (None, n)
    w_l = (s_u - s_t) / K.maximum(K.epsilon(), u_c_flat - t_value)  # (None, n)
    b_l = -w_l * u_c_flat + s_u  # func(u_c_flat)

    return [w_l, b_l]


def get_linear_hull_s_shape(x, func=K.sigmoid, f_prime=sigmoid_prime, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """
    Computing the linear hull of shape functions  given the pre activation neurons

    :param x: list of input tensors
    :param func: the function (sigmoid, tanh, softsign...)
    :param f_prime: the derivative of the function (sigmoid_prime...)
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: the updated list of tensors
    """

    if mode == F_IBP.name:
        y, x_0, u_c, l_c = x[:4]
    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
    if mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = x[:6]
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)

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
    coeff = (s_u - s_l) / K.maximum(K.epsilon(), u_c_flat - l_c_flat)
    alpha_u_0 = K.switch(greater_equal(s_u_prime, coeff), 1.0 + 0 * u_c_flat, 0 * u_c_flat)  # (None, n)
    alpha_u_1 = (1 - alpha_u_0) * ((K.sign(l_c_flat) + 1.0) / 2.0)

    w_u_0 = coeff
    b_u_0 = -w_u_0 * l_c_flat + s_l

    w_u_1 = 0 * u_c_flat
    b_u_1 = s_u

    w_u_2, b_u_2 = get_t_upper(u_c_flat, l_c_flat, s_l, func=func, f_prime=f_prime)

    w_u_ = K.reshape(alpha_u_0 * w_u_0 + alpha_u_1 * w_u_1 + (1 - alpha_u_0 - alpha_u_1) * w_u_2, [-1] + shape)
    b_u_ = K.reshape(alpha_u_0 * b_u_0 + alpha_u_1 * b_u_1 + (1 - alpha_u_0 - alpha_u_1) * b_u_2, [-1] + shape)
    # w_u_ = K.reshape(w_u_, [-1]+shape)
    # b_u_ = K.reshape(b_u_, [-1]+shape)

    # linear hull

    # case 0:
    alpha_l_0 = K.switch(greater_equal(s_l_prime, coeff), 1.0 + 0 * l_c_flat, 0 * l_c_flat)  # (None, n)
    alpha_l_1 = (1 - alpha_l_0) * ((K.sign(-u_c_flat) + 1.0) / 2.0)

    w_l_0 = coeff
    b_l_0 = -w_l_0 * u_c_flat + s_u

    w_l_1 = 0 * u_c_flat
    b_l_1 = s_l

    w_l_2, b_l_2 = get_t_lower(u_c_flat, l_c_flat, s_u, func=func, f_prime=f_prime)

    w_l_ = K.reshape(alpha_l_0 * w_l_0 + alpha_l_1 * w_l_1 + (1 - alpha_l_0 - alpha_l_1) * w_l_2, [-1] + shape)
    b_l_ = K.reshape(alpha_l_0 * b_l_0 + alpha_l_1 * b_l_1 + (1 - alpha_l_0 - alpha_l_1) * b_l_2, [-1] + shape)

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
        b_l_ = alpha_l_ * b_l_ + (1.0 - alpha_l_) * op_flat(K.maximum(l_c, 0.0))

        w_u_ = K.reshape(w_u_, shape_w)
        w_l_ = K.reshape(w_l_, shape_w)
        b_u_ = K.reshape(b_u_, shape_b)
        b_l_ = K.reshape(b_l_, shape_b)

    return [w_u_, b_u_, w_l_, b_l_]  # what the hell !!!!


def minus(inputs, mode=F_HYBRID.name, dc_decomp=False):
    """
    LiRPA implementation of minus(x)=-x.
    :param inputs:
    :param mode:
    :return:
    """

    if mode == F_IBP.name:
        y, x, u, l = inputs[:4]
    if mode == F_FORWARD.name:
        y, x, w_u, b_u, w_l, b_l = inputs[:6]
    if mode == F_HYBRID.name:
        y, x, u, w_u, b_u, l, w_l, b_l = inputs[:8]

    if dc_decomp:
        h, g = inputs[-2:]

    y_ = -y

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_ = -l
        l_ = -u

    if mode in [F_FORWARD.name, F_HYBRID.name]:
        w_u_ = -w_l
        b_u_ = -b_l
        w_l_ = -w_u
        b_l_ = -b_u

    output = [y_, x]
    if mode == F_IBP.name:
        output += [u_, l_]
    if mode == F_FORWARD.name:
        output += [w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        output += [u_, w_u_, b_u_, l_, w_l_, b_l_]

    if dc_decomp:
        output += [-g, -h]

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

    if mode == F_IBP.name:
        y0, x, u0, l0 = inputs_0[:4]
        y1, _, u1, l1 = inputs_1[:4]
    if mode == F_FORWARD.name:
        y0, x, w_u0, b_u0, w_l0, b_l0 = inputs_0[:6]
        y1, _, w_u1, b_u1, w_l1, b_l1 = inputs_1[:6]
    if mode == F_HYBRID.name:
        y0, x, u0, w_u0, b_u0, l0, w_l0, b_l0 = inputs_0[:8]
        y1, _, u1, w_u1, b_u1, l1, w_l1, b_l1 = inputs_1[:8]

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
        inputs_0_ = [y0 * y1, x, upper_0, w_0_u, b_u_0, lower_0, w_0_l, b_l_0]
        inputs_1_ = [y0 * y1, x, upper_1, w_1_u, b_u_1, lower_1, w_1_l, b_l_1]

    if mode == F_FORWARD.name:
        inputs_0_ = [y0 * y1, x, w_0_u, b_u_0, w_0_l, b_l_0]
        inputs_1_ = [y0 * y1, x, w_1_u, b_u_1, w_1_l, b_l_1]

    if mode == F_IBP.name:
        output = [y0 * y1, x, K.minimum(upper_0, upper_1), K.maximum(lower_0, lower_1)]
    else:
        output_upper = minimum(inputs_0_, inputs_1_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
        output_lower = maximum(inputs_0_, inputs_1_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)

        if mode == F_FORWARD.name:
            _, _, w_u_, b_u_, _, _ = output_upper
            _, _, _, _, w_l_, b_l_ = output_upper

            output = [y0 * y1, x, w_u_, b_u_, w_l_, b_l_]

        if mode == F_HYBRID.name:
            _, _, u_, w_u_, b_u_, _, _, _ = output_upper
            _, _, _, _, _, l_, w_l_, b_l_ = output_upper

            output = [y0 * y1, x, u_, w_u_, b_u_, l_, w_l_, b_l_]

    return output


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

    if mode == F_IBP.name:
        return [
            K.permute_dimensions(x[0], index),
            x[1],
            K.permute_dimensions(x[2], index),
            K.permute_dimensions(x[3], index),
        ]

    index_w = np.arange(len(x[0].shape) + 1)
    index_w = np.insert(np.delete(index_w, axis), axis_perm + 1, axis)

    if mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = x
        return [
            K.permute_dimensions(y, index),
            x_0,
            K.permute_dimensions(w_u, index_w),
            K.permute_dimensions(b_u, index),
            K.permute_dimensions(w_l, index_w),
            K.permute_dimensions(b_l, index),
        ]

    if mode == F_HYBRID.name:
        y, x_0, u, w_u, b_u, l, w_l, b_l = x

        return [
            K.permute_dimensions(y, index),
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

    if mode == F_IBP.name:
        y, x, u, l = inputs
    if mode == F_FORWARD.name:
        y, x, w_u, b_u, w_l, b_l = inputs
    if mode == F_HYBRID.name:
        y, x, u, w_u, b_u, l, w_l, b_l = inputs

    for _ in range(n):
        y = K.expand_dims(y, axis)

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
        output = [y, x, u, l]
    if mode == F_FORWARD.name:
        output = [y, x, w_u, b_u, w_l, b_l]
    if mode == F_HYBRID.name:
        output = [y, x, u, w_u, b_u, l, w_l, b_l]

    return output


def split(input_, axis=-1, mode=F_HYBRID.name):
    """
    LiRPA implementation of split

    :param input_:
    :param axis:
    :param mode:
    :return:
    """
    if mode == F_IBP.name:
        y_, x_, u_, l_ = input_

    if mode == F_HYBRID.name:
        y_, x_, u_, w_u_, b_u_, l_, w_l_, b_l_ = input_

    if mode == F_FORWARD.name:
        y_, x_, w_u_, b_u_, w_l_, b_l_ = input_

    y_list = tf.split(y_, 1, axis=axis)

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_list = tf.split(u_, 1, axis=axis)
        l_list = tf.split(l_, 1, axis=axis)

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        b_u_list = tf.split(b_u_, 1, axis=axis)
        b_l_list = tf.split(b_l_, 1, axis=axis)

        if axis != -1:
            axis += 1
        w_u_list = tf.split(w_u_, 1, axis=axis)
        w_l_list = tf.split(w_l_, 1, axis=axis)

    n = len(y_list)
    if mode == F_IBP.name:
        outputs = [[y_list[i], x_, u_list[i], l_list[i]] for i in range(n)]

    if mode == F_FORWARD.name:
        outputs = [[y_list[i], x_, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]

    if mode == F_HYBRID.name:
        outputs = [
            [y_list[i], x_, u_list[i], w_u_list[i], b_u_list[i], l_list[i], w_l_list[i], b_l_list[i]] for i in range(n)
        ]

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

    if mode == F_IBP.name:
        y, x_0, u_c, l_c = input_

    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = input_

    if mode == F_FORWARD.name:
        y_, x_0, w_u, b_u, w_l, b_l = input_

    if axis == -1:
        n = y.shape[-1]
        axis = len(y.shape) - 1
    else:
        n = y.shape[axis]

    y_ = tf.sort(y, axis=axis)

    # what about splitting elements
    op = lambda x: tf.split(x, n, axis=axis)
    y_list = op(y)

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
            return [y_list[i], x_0, u_c_list[i], l_c_list[i]]
        if mode == F_FORWARD.name:
            return [y_list[i], x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
        if mode == F_HYBRID.name:
            return [y_list[i], x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]

    def set_input(input_, mode, i):
        if mode == F_IBP.name:
            y_i, _, u_i, l_i = input_
        if mode == F_FORWARD.name:
            y_i, _, w_u_i, b_u_i, w_l_i, b_l_i = input_
        if mode == F_HYBRID.name:
            y_i, _, u_i, w_u_i, b_u_i, l_i, w_l_i, b_l_i = input_

        y_list[i] = y_i
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
        output = [y_, x_0, u_c_, l_c_]
    if mode == F_FORWARD.name:
        output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

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

    if mode == F_IBP.name:
        y, x_0, u_c, l_c = inputs_
    if mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = inputs_
    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs_

    y_ = 1 / y
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
def min_(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, axis=-1):
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
        max_(minus(x, mode=mode), dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode, axis=axis), mode=mode
    )


def expand_dims(inputs_, dc_decomp=False, mode=F_HYBRID.name, axis=-1):

    if mode == F_IBP.name:
        y, x_0, u_c, l_c = inputs_[:4]
    if mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = inputs_[:6]
    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs_[:8]

    if dc_decomp:
        h, g = inputs_[-2:]

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        if axis == -1:
            axis_w = axis
        else:
            axis_w = axis + 1

    op = lambda t: K.expand_dims(t, axis)

    y_ = op(y)
    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = op(u_c)
        l_c_ = op(l_c)

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        b_u_ = op(b_u)
        b_l_ = op(b_l)
        w_u_ = K.expand_dims(w_u, axis_w)
        w_l_ = K.expand_dims(w_l, axis_w)

    if mode == F_IBP.name:
        output = [y_, x_0, u_c_, l_c_]

    if mode == F_FORWARD.name:
        output = [y, x_0, w_u_, b_u_, w_l_, b_l_]

    if mode == F_HYBRID.name:
        output = [y, x_0, u_c_, w_u_, l_c_, w_l_, b_l_]

    return output
